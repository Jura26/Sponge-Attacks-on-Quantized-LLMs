import argparse
import glob
import logging
import os
import re
from types import SimpleNamespace

import torch

from model_catalog import (
    GGUF_VARIANT_TO_MODE,
    MODEL_FAMILIES,
    resolve_gguf_path_for_variant,
    resolve_gptq_repo,
    resolve_hf_local_path,
    resolve_hf_repo,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


GGUF_VARIANT_TOKENS = {
    "gguf-f16": ["f16", "fp16"],
    "gguf-q8": ["q8_0", "q8"],
    "gguf-q6": ["q6_k", "q6_k_l", "q6"],
    "gguf-q5": ["q5_k_m", "q5_k_s", "q5_1", "q5_0", "q5"],
    "gguf-q4": ["q4_k_m", "q4_k_s", "q4_1", "q4_0", "q4"],
    "gguf-q3": ["q3_k_l", "q3_k_m", "q3_k_s", "q3_0", "q3"],
    "gguf-q2": ["q2_k", "q2_0", "q2"],
}

_LAST_GGUF_SELECTION = {
    "model_id": None,
    "quant_mode": None,
    "path": None,
}


class _InputBatch(dict):
    """Minimal tensor-batch wrapper with `.to()` for attack code compatibility."""

    def to(self, device):
        moved = _InputBatch()
        for key, value in self.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def tokenize_for_attack(tokenizer, text, device, *, truncation=False, max_length=None):
    """Tokenize text to `input_ids` on `device` (HF + GGUF adapters)."""
    kwargs = {"return_tensors": "pt"}
    if truncation:
        kwargs["truncation"] = True
    if max_length is not None:
        kwargs["max_length"] = max_length

    batch = tokenizer(text, **kwargs)
    if hasattr(batch, "to"):
        batch = batch.to(device)
    elif isinstance(batch, dict):
        batch = _InputBatch({
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        })
    else:
        batch = _InputBatch(input_ids=batch["input_ids"].to(device))

    return batch


class GGUFTokenizerAdapter:
    """Tokenizer-like adapter for llama.cpp backends."""

    def __init__(self, llm):
        self.llm = llm
        self.eos_token_id = 2
        self.pad_token_id = 2
        self.vocab_size = int(getattr(llm, "n_vocab", lambda: 32000)())

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False)

    def decode(self, tokens, skip_special_tokens=True):
        _ = skip_special_tokens
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        b = self.llm.detokenize(tokens)
        return b.decode("utf-8", errors="ignore")

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
        _ = return_tensors
        toks = self.encode(text, add_special_tokens=False)
        if truncation and max_length is not None:
            toks = toks[:max_length]
        t = torch.tensor([toks], dtype=torch.long)
        return _InputBatch(input_ids=t)


class GGUFModelAdapter:
    """Model-like adapter exposing `generate()` and prefill call for llama.cpp."""

    def __init__(self, llm, n_ctx=4096):
        self.llm = llm
        # Keep CPU tensor semantics for adapter compatibility.
        # GPU offload is managed inside llama.cpp by n_gpu_layers.
        self.device = "cpu"
        self.config = SimpleNamespace(max_position_embeddings=int(n_ctx), n_positions=int(n_ctx))

    def eval(self):
        return self

    def generate(self, input_ids=None, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, **_kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")

        if hasattr(input_ids, "tolist"):
            in_toks = input_ids.tolist()[0]
        else:
            in_toks = list(input_ids[0])

        prompt = self.llm.detokenize(in_toks).decode("utf-8", errors="ignore")
        stop = _kwargs.get("stop")
        out = self.llm.create_completion(
            prompt=prompt,
            max_tokens=int(max_new_tokens),
            temperature=float(temperature if do_sample else 0.0),
            top_p=float(top_p),
            stop=stop,
            echo=False,
        )
        gen_text = out["choices"][0].get("text", "")
        gen_toks = self.llm.tokenize(gen_text.encode("utf-8"), add_bos=False)
        merged = in_toks + gen_toks
        return torch.tensor([merged], dtype=torch.long)

    def __call__(self, input_ids=None, use_cache=True, return_dict=True, **_kwargs):
        _ = use_cache
        if input_ids is None:
            raise ValueError("input_ids is required")
        toks = input_ids.tolist()[0]

        try:
            self.llm.reset()
            self.llm.eval(toks)
        except Exception:
            prompt = self.llm.detokenize(toks).decode("utf-8", errors="ignore")
            self.llm.create_completion(prompt=prompt, max_tokens=1, temperature=0.0, top_p=1.0, echo=True)

        if return_dict:
            return {"last_hidden_state": None}
        return None


def _quant_bits_from_name(path: str) -> int:
    name_l = os.path.basename(path).lower()
    if "f16" in name_l or "fp16" in name_l:
        return 16
    m = re.search(r"q([2-8])(?:[_\\.-]|$)", name_l)
    if m:
        return int(m.group(1))
    return 0


def _quant_mode_from_path(path: str) -> str | None:
    bits = _quant_bits_from_name(path)
    if bits == 16:
        return "gguf-f16"
    if bits in (2, 3, 4, 5, 6, 8):
        return f"gguf-q{bits}"
    return None


def _family_hint_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    stem = re.sub(r"[-_.]?(f16|fp16|fp32).*$", "", stem)
    stem = re.sub(r"[-_.]?q[2-8].*$", "", stem)
    return stem.strip("-_. ")


_HF_TOKEN_CACHE = {"checked": False, "value": None}


def _valid_hf_token():
    if _HF_TOKEN_CACHE["checked"]:
        return _HF_TOKEN_CACHE["value"]

    _HF_TOKEN_CACHE["checked"] = True
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token in {"hf_", "your_token", "hf_your_token_here"}:
        return None

    try:
        from huggingface_hub import HfApi
        HfApi().whoami(token=token)
        _HF_TOKEN_CACHE["value"] = token
        return token
    except Exception:
        logger.warning("⚠️  HF_TOKEN is set but not valid; ignoring it (clear backend/.env to silence).")
        return None


def gptq_available() -> bool:
    try:
        import gptqmodel  # noqa: F401
        return True
    except ImportError:
        return False


def hf_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _quant_mode_to_gguf_variant(quant_mode: str) -> str:
    inv = {v: k for k, v in GGUF_VARIANT_TO_MODE.items()}
    return inv.get(quant_mode, "q4_k_m")


def _resolve_gptq_model_id(model_id: str) -> str:
    """Map family id or HF repo to a GPTQ checkpoint."""
    model_id = (model_id or "").strip()
    if model_id.startswith("gguf:") or model_id.lower().endswith(".gguf"):
        raise RuntimeError(
            f"GPTQ backend cannot load a GGUF path ({model_id!r}). "
            "Select backend=GPTQ with a model family, not a .gguf file."
        )
    if os.path.isabs(model_id) or model_id.startswith(("./", "../")):
        raise RuntimeError(f"GPTQ backend cannot load local path {model_id!r}")

    if model_id in MODEL_FAMILIES:
        return resolve_gptq_repo(model_id)

    # Hugging Face repo id (namespace/name)
    if "/" in model_id and not model_id.startswith("/"):
        return model_id

    env_default = os.environ.get("SPONGE_GPTQ_MODEL_ID", "").strip()
    if env_default:
        return env_default

    raise RuntimeError(
        f"No GPTQ model for id '{model_id}'. Use a catalog family id "
        "(mistral7b, gpt2, opt-6.7b) or set SPONGE_GPTQ_MODEL_* in .env."
    )


def _normalize_quant_mode(quantize) -> str:
    """Normalize quant mode for GGUF (llama.cpp) or GPTQ (GPTQModel) backends."""
    if quantize is False or quantize is None:
        return "gguf-f16"
    if quantize is True:
        return "gguf-q4"

    mode = str(quantize).strip().lower()
    aliases = {
        "none": "gguf-f16",
        "full": "gguf-f16",
        "fp16": "gguf-f16",
        "fp32": "gguf-f16",
        "no": "gguf-f16",
        "off": "gguf-f16",
        "gguf": "gguf-q4",
        "llamacpp": "gguf-q4",
        "llama.cpp": "gguf-q4",
        "gguf-llamacpp": "gguf-q4",
        "gguf-f16": "gguf-f16",
        "gguf-q8": "gguf-q8",
        "gguf-q6": "gguf-q6",
        "gguf-q5": "gguf-q5",
        "gguf-q4": "gguf-q4",
        "gguf-q3": "gguf-q3",
        "gguf-q2": "gguf-q2",
        "gptq": "gptq",
        "gptq-int4": "gptq",
        "gptq-4bit": "gptq",
        "hf": "hf-fp16",
        "hf-fp16": "hf-fp16",
        "transformers-fp16": "hf-fp16",
        # Legacy PyTorch quant names → GGUF compare path
        "bnb-nf4": "gguf-q4",
        "bnb-fp4": "gguf-q4",
        "bnb-int8": "gguf-q8",
        "int8-cpu": "gguf-q8",
    }
    return aliases.get(mode, "gguf-f16")


def _resolve_hf_model_hint(model_id: str) -> tuple[str, str | None]:
    """Return (model_ref, repo_id) for HF loading with local-only preference."""
    model_id = (model_id or "").strip()
    if not model_id:
        raise RuntimeError("HF backend needs a model id or family key.")

    if os.path.isabs(model_id) or model_id.startswith(("./", "../")):
        if os.path.isdir(model_id):
            return model_id, None
        raise RuntimeError(f"HF backend cannot find local path {model_id!r}")

    if model_id in MODEL_FAMILIES:
        repo = resolve_hf_repo(model_id)
        local_path = resolve_hf_local_path(model_id)
        return (local_path or repo), repo

    if "/" in model_id and not model_id.startswith("/"):
        hf_dir = os.environ.get("SPONGE_HF_DIR", "").strip()
        if hf_dir:
            candidate = os.path.join(hf_dir, model_id)
            if os.path.isdir(candidate):
                return candidate, model_id
        return model_id, model_id

    raise RuntimeError(
        f"HF backend cannot resolve model id '{model_id}'. "
        "Use a family id (Llama3, Qwen, Hunyuan) or a HF repo id."
    )


def _load_hf_fp16_model(model_id: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install it in the backend venv to use HF FP16."
        ) from exc

    model_ref, repo = _resolve_hf_model_hint(model_id)
    trust_remote = os.environ.get("SPONGE_HF_TRUST_REMOTE_CODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    logger.info(f"⚙️  Loading HF FP16 via transformers: {model_ref}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=trust_remote,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            use_fast=False,
            local_files_only=True,
            trust_remote_code=trust_remote,
        )
    except Exception as exc:
        repo_hint = repo or model_ref
        raise RuntimeError(
            f"HF FP16 model '{repo_hint}' not available locally. "
            "Run scripts/download_hf_models.py to pre-download it."
        ) from exc

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply optional context cap
    max_context_override = int(os.environ.get("SPONGE_MAX_CONTEXT", "0"))
    if max_context_override > 0:
        native_ctx = getattr(model.config, "max_position_embeddings", None)

        if native_ctx is not None:
            model.config.max_position_embeddings = min(
                native_ctx,
                max_context_override,
            )
            logger.info(
                f"📏 Context limited to {model.config.max_position_embeddings} "
                f"(native={native_ctx})"
            )
        else:
            model.config.max_position_embeddings = max_context_override
            logger.info(
                f"📏 Context set to {model.config.max_position_embeddings}"
            )

    device = "cpu"
    try:
        param = next(model.parameters())
        if param.device.type == "cuda":
            device = "cuda"
    except StopIteration:
        pass

    logger.info("✅ Model ready — HF FP16 backend (transformers).")
    return tokenizer, model, device, "hf-fp16"


def resolve_gguf_variant_path(model_id: str, quant_mode: str) -> str:
    """Resolve GGUF file path for a model id + quant mode."""
    gguf_mode = quant_mode if quant_mode in GGUF_VARIANT_TOKENS else "gguf-q4"

    if model_id in MODEL_FAMILIES:
        variant = _quant_mode_to_gguf_variant(gguf_mode)
        path = resolve_gguf_path_for_variant(model_id, variant)
        if path:
            return path
        family_label = MODEL_FAMILIES[model_id]["label"]
        raise RuntimeError(
            f"No GGUF file for {family_label} ({variant}). Check SPONGE_GGUF_DIR."
        )

    if model_id.startswith("gguf:"):
        return model_id.split("gguf:", 1)[1].strip()
    if model_id.lower().endswith(".gguf"):
        return model_id

    default_path = os.environ.get("SPONGE_GGUF_PATH", "").strip()
    gguf_dir = os.environ.get("SPONGE_GGUF_DIR", "").strip()
    if not gguf_dir and default_path:
        gguf_dir = os.path.dirname(default_path)

    mode_tokens = GGUF_VARIANT_TOKENS.get(gguf_mode, [])
    basename_filter = os.environ.get("SPONGE_GGUF_BASENAME", "").strip().lower()
    if not basename_filter and default_path:
        basename_filter = _family_hint_from_path(default_path)
    if not basename_filter:
        model_hint = os.path.basename(model_id).strip().lower()
        if model_hint and model_hint != model_id.lower():
            basename_filter = model_hint
        elif model_hint:
            basename_filter = model_hint

    if gguf_dir and os.path.isdir(gguf_dir):
        candidates = sorted(glob.glob(os.path.join(gguf_dir, "*.gguf")))
        if basename_filter:
            filtered = [p for p in candidates if basename_filter in os.path.basename(p).lower()]
            if filtered:
                candidates = filtered
            else:
                return default_path

        if not candidates:
            return default_path

        for token in mode_tokens:
            token_l = token.lower()
            for path in candidates:
                if token_l in os.path.basename(path).lower():
                    return path

        if gguf_mode == "gguf-f16":
            return min(
                candidates,
                key=lambda p: (
                    0 if ("f16" in os.path.basename(p).lower() or "fp16" in os.path.basename(p).lower()) else 1,
                    -_quant_bits_from_name(p),
                ),
            )

        target_bits = {
            "gguf-q8": 8,
            "gguf-q6": 6,
            "gguf-q5": 5,
            "gguf-q4": 4,
            "gguf-q3": 3,
            "gguf-q2": 2,
        }.get(gguf_mode, 4)

        return min(
            candidates,
            key=lambda p: (
                abs(_quant_bits_from_name(p) - target_bits) if _quant_bits_from_name(p) else 99,
                -_quant_bits_from_name(p),
            ),
        )

    return default_path


def get_last_gguf_selection() -> dict:
    return dict(_LAST_GGUF_SELECTION)


def _load_gguf_llamacpp(model_id: str, quant_mode: str):
    try:
        from llama_cpp import Llama, llama_supports_gpu_offload
    except Exception as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install it in the backend venv to use GGUF modes."
        ) from exc

    gguf_path = resolve_gguf_variant_path(model_id, quant_mode)

    if not gguf_path:
        raise RuntimeError(
            "GGUF path not provided. Set model_id='gguf:/absolute/path/model.gguf' or set "
            "SPONGE_GGUF_PATH / SPONGE_GGUF_DIR in backend/.env"
        )
    if not os.path.isfile(gguf_path):
        raise RuntimeError(f"GGUF file not found: {gguf_path}")

    detected_quant_mode = _quant_mode_from_path(gguf_path)
    if detected_quant_mode and detected_quant_mode != quant_mode:
        quant_mode = detected_quant_mode

    _LAST_GGUF_SELECTION["model_id"] = model_id
    _LAST_GGUF_SELECTION["quant_mode"] = quant_mode
    _LAST_GGUF_SELECTION["path"] = gguf_path

    n_ctx = int(os.environ.get("SPONGE_GGUF_CTX", "4096"))
    n_threads = int(os.environ.get("SPONGE_GGUF_THREADS", str(max(1, (os.cpu_count() or 8) // 2))))
    n_gpu_layers = int(os.environ.get("SPONGE_GGUF_GPU_LAYERS", "-1"))

    supports_offload = bool(llama_supports_gpu_offload())
    if n_gpu_layers != 0 and not supports_offload:
        logger.warning("⚠️  llama.cpp build has no GPU offload support; running GGUF on CPU.")
        n_gpu_layers = 0

    logger.info(f"⚙️  Loading GGUF via llama.cpp: {gguf_path}")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    tokenizer = GGUFTokenizerAdapter(llm)
    model = GGUFModelAdapter(llm, n_ctx=n_ctx)
    quant_label = quant_mode
    device = "cuda" if supports_offload and n_gpu_layers != 0 else "cpu"

    if device == "cuda":
        logger.info("✅ Model ready — GGUF backend (GPU offload active).")
    else:
        logger.info("✅ Model ready — GGUF backend (CPU mode).")

    return tokenizer, model, device, quant_label


def _load_gptq_model(model_id: str):
    try:
        from gptqmodel import GPTQModel
    except ImportError as exc:
        raise RuntimeError(
            "GPTQModel is not installed. On ROCm run: bash scripts/install_gptq_rocm.sh"
        ) from exc

    # Hunyuan GPTQ ships a Llama-like module layout but uses a custom model_type.
    # GPTQModel does not register it by default, so we map it to LlamaQModel.
    try:
        from gptqmodel.models import auto as gptq_auto
        from gptqmodel.models.definitions.llama import LlamaQModel

        if "hunyuan_v1_dense" not in gptq_auto.MODEL_MAP:
            gptq_auto.MODEL_MAP["hunyuan_v1_dense"] = LlamaQModel
        if hasattr(gptq_auto, "SUPPORTED_MODELS") and "hunyuan_v1_dense" not in gptq_auto.SUPPORTED_MODELS:
            gptq_auto.SUPPORTED_MODELS.append("hunyuan_v1_dense")
    except Exception:
        pass

    hf_id = _resolve_gptq_model_id(model_id)
    gptq_dir = os.environ.get("SPONGE_GPTQ_DIR", "").strip()
    local_path = None
    if gptq_dir:
        direct_cfg = os.path.join(gptq_dir, "config.json")
        if os.path.isfile(direct_cfg):
            local_path = gptq_dir
        else:
            candidate = os.path.join(gptq_dir, *hf_id.split("/"))
            if os.path.isdir(candidate):
                local_path = candidate
    if local_path:
        hf_id = local_path
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    token = _valid_hf_token()
    load_kwargs = {"device": device}
    trust_remote = os.environ.get("SPONGE_GPTQ_TRUST_REMOTE_CODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if trust_remote:
        load_kwargs["trust_remote_code"] = True
    if local_path:
        load_kwargs["local_files_only"] = True
    if token:
        load_kwargs["token"] = token

    logger.info(f"⚙️  Loading GPTQ via GPTQModel: {hf_id} ({device})")
    wrapper = GPTQModel.load(hf_id, **load_kwargs)
    tokenizer = wrapper.tokenizer
    model = wrapper.model
    model.eval()

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quant_label = "gptq-int4"
    qcfg = getattr(model.config, "quantization_config", None)
    if qcfg is not None:
        bits = qcfg.get("bits", 4) if isinstance(qcfg, dict) else getattr(qcfg, "bits", 4)
        quant_label = f"gptq-{bits}bit"

    MAX_CONTEXT_OVERRIDE = int(os.environ.get("SPONGE_MAX_CONTEXT", 0))
    if MAX_CONTEXT_OVERRIDE > 0:
        model.config.max_position_embeddings = MAX_CONTEXT_OVERRIDE

    backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
    logger.info(f"✅ Model ready — GPTQ backend on {backend} ({quant_label}).")
    return tokenizer, model, "cuda" if device.startswith("cuda") else "cpu", quant_label


def cleanup_model(model, tokenizer=None):
    """Release references and clear caches between runs."""
    import gc

    try:
        llm = getattr(model, "llm", None)
        if llm is not None and hasattr(llm, "reset"):
            llm.reset()
    except Exception:
        pass

    del model
    if tokenizer is not None:
        del tokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(model_id: str, quantize=False):
    """Load GGUF (llama.cpp) or GPTQ (GPTQModel) backends.

    GGUF modes: gguf-f16, gguf-q8 … gguf-q2
    GPTQ modes: gptq, gptq-int4 (Hugging Face GPTQ checkpoint)
    """
    quant_mode = _normalize_quant_mode(quantize)
    if quant_mode == "gptq":
        return _load_gptq_model(model_id)
    if quant_mode == "hf-fp16":
        return _load_hf_fp16_model(model_id)
    supported_modes = set(GGUF_VARIANT_TOKENS.keys())
    if quant_mode not in supported_modes:
        quant_mode = "gguf-f16"
    return _load_gguf_llamacpp(model_id, quant_mode)


def generate_text(model_id: str, prompt: str, max_new_tokens: int = -1):
    tokenizer, model, _device, _quant_label = load_model_and_tokenizer(model_id)

    logger.info("🔄 Generating response...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    if max_new_tokens == -1:
        model_max_length = getattr(model.config, "max_position_embeddings", None)
        if model_max_length is None:
            model_max_length = getattr(model.config, "n_positions", 4096)
        max_new_tokens = max(1, model_max_length - input_len)
        logger.info(f"✨ Auto-setting max tokens to: {max_new_tokens}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    print("\n" + "=" * 40)
    print(f"📝 Prompt: {prompt}")
    print("-" * 40)
    print(f"🤖 Response:\n{generated_text}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local GGUF loader and text generator.")
    parser.add_argument("model_id", type=str, help="GGUF file path or model id hint")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=-1, help="Max new tokens to generate (-1 for auto)")

    args = parser.parse_args()
    generate_text(args.model_id, args.prompt, args.max_tokens)
