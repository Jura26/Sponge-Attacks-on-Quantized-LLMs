import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# auto-gptq requires QuantizeConfig and FORMAT to be importable in the global
# namespace when deserializing GPTQ model configs — import and inject them here.
import builtins

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig as QuantizeConfig  # noqa: F401
except ImportError:
    try:
        from auto_gptq import QuantizeConfig  # noqa: F401
    except ImportError:
        try:
            from transformers import GPTQConfig as QuantizeConfig  # noqa: F401
        except ImportError:
            pass

try:
    from auto_gptq.quantize import FORMAT  # noqa: F401
except ImportError:
    try:
        from auto_gptq import FORMAT  # noqa: F401
    except ImportError:
        pass

# Inject into builtins so pickle/deserialization can always find them
for _name in ("QuantizeConfig", "FORMAT"):
    if _name in dir() and not hasattr(builtins, _name):
        setattr(builtins, _name, eval(_name))

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def cleanup_model(model, tokenizer=None):
    """Aggressively free a model from GPU VRAM.

    Handles models loaded with device_map='auto' (accelerate dispatch hooks)
    which prevent normal .cpu() / del from releasing VRAM.
    """
    import gc

    if torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"🧹 VRAM before cleanup: {allocated_before:.2f} GB allocated")

    # 1. Remove accelerate dispatch hooks (they hold GPU tensor references)
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    except (ImportError, Exception):
        pass

    # 2. Move every parameter/buffer to CPU individually
    #    (model.cpu() can fail on dispatched models)
    try:
        for param in model.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad = param.grad.cpu()
        for buf in model.buffers():
            buf.data = buf.data.cpu()
    except Exception:
        pass

    # 3. Try the normal .cpu() as well
    try:
        model.cpu()
    except Exception:
        pass

    # 4. Delete references
    del model
    if tokenizer is not None:
        del tokenizer

    # 5. Force garbage collection
    gc.collect()

    # 6. Release VRAM back to the OS
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"🧹 VRAM after cleanup: {allocated_after:.2f} GB allocated")


def _is_model_cached(model_id: str, hf_token=None) -> bool:
    """Check if a model exists in the HuggingFace cache without making network requests."""
    from huggingface_hub import try_to_load_from_cache, scan_cache_dir
    try:
        result = try_to_load_from_cache(model_id, "config.json", token=hf_token)
        # Returns None if not cached, a path string if found
        if result is not None and isinstance(result, str):
            return True
    except Exception:
        pass

    # Fallback: scan cache directory
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                return True
    except Exception:
        pass

    return False


def load_model_and_tokenizer(model_id: str):
    """
    Load a HuggingFace causal-LM and its tokenizer.

    Supports regular models AND pre-quantized models (GPTQ, AWQ, etc.).
    Quantization method is auto-detected from the model's config.

    Returns:
        (tokenizer, model, device, quant_label)
    """

    # --- Log VRAM state before loading ---
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"🔧 VRAM state: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Log GPU/ROCm info
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        hip_version = getattr(torch.version, 'hip', None)
        if hip_version:
            logger.info(f"🎮 GPU: {gpu_name} (ROCm/HIP {hip_version})")
        else:
            logger.info(f"🎮 GPU: {gpu_name} (CUDA {torch.version.cuda})")

    logger.info(f"🔄 Checking model {model_id}...")

    # Use HF_TOKEN env var for authenticated (faster) downloads
    hf_token = os.environ.get("HF_TOKEN")

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if device == "cuda" else None,
        "token": hf_token,
    }

    # Detect ROCm and pre-configure GPTQ to avoid broken ExLlama kernels.
    # ExLlama/ExLlamaV2 are NVIDIA-only CUDA kernels; on AMD/ROCm they silently
    # fail and fall back to extremely slow pure-PyTorch dequantization.
    is_rocm = getattr(torch.version, 'hip', None) is not None
    if is_rocm:
        try:
            from transformers import AutoConfig
            remote_config = AutoConfig.from_pretrained(
                model_id, token=hf_token,
                **({"local_files_only": True} if _is_model_cached(model_id, hf_token) else {})
            )
            qcfg = getattr(remote_config, "quantization_config", None)
            if qcfg:
                qdict = qcfg if isinstance(qcfg, dict) else qcfg.to_dict()
                if qdict.get("quant_method") in ("gptq", "awq"):
                    from transformers import GPTQConfig
                    bits = qdict.get("bits", 4)
                    group_size = qdict.get("group_size", 128)
                    desc_act = qdict.get("desc_act", False)
                    logger.info(f"⚙️ ROCm detected — disabling ExLlama kernels for GPTQ model")
                    load_kwargs["quantization_config"] = GPTQConfig(
                        bits=bits,
                        group_size=group_size,
                        desc_act=desc_act,
                        disable_exllama=True,
                        use_cuda_fp16=True,
                    )
        except Exception as e:
            logger.warning(f"⚠️ Could not pre-configure GPTQ for ROCm: {e}")

    if _is_model_cached(model_id, hf_token):
        logger.info(f"✅ Found {model_id} in local cache.")
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, **load_kwargs
        )
    else:
        if not hf_token:
            logger.warning("⚠️ No HF_TOKEN set — downloads will be slow! Set HF_TOKEN env var for faster downloads.")
        logger.info(f"⬇️ Model {model_id} not found locally. Downloading...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        logger.info(f"✅ Download complete and model loaded.")

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device == "cpu":
        model.to(device)

    # Auto-detect quantization from model config (GPTQ, AWQ, etc.)
    quant_config = getattr(model.config, "quantization_config", None)
    if quant_config:
        if isinstance(quant_config, dict):
            quant_method = quant_config.get("quant_method", "quantized")
            bits = quant_config.get("bits", "?")
        else:
            quant_method = getattr(quant_config, "quant_method", "quantized")
            bits = getattr(quant_config, "bits", "?")
        quant_label = f"{quant_method}-{bits}bit"
        logger.info(f"✅ Model ready — {quant_method} {bits}-bit quantization detected.")
    else:
        quant_label = "fp16" if device == "cuda" else "fp32"
        logger.info(f"✅ Model ready on {device.upper()} ({quant_label}).")

    model.eval()
    return tokenizer, model, device, quant_label

def generate_text(model_id: str, prompt: str, max_new_tokens: int = -1):
    """
    Loads model and generates text based on prompt.
    """
    tokenizer, model, device, _quant_label = load_model_and_tokenizer(model_id)

    logger.info("🔄 Generating response...")
    
    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Determine max tokens if set to auto (-1)
    if max_new_tokens == -1:
        # Try to find model's max context length
        model_max_length = getattr(model.config, "max_position_embeddings", None)
        if model_max_length is None:
             model_max_length = getattr(model.config, "n_positions", None)
        
        # Fallback if unknown (common for some architectures)
        if model_max_length is None:
             model_max_length = 4096 # Safe modern default
             logger.warning(f"⚠️ Could not detect model context length. Defaulting to {model_max_length}.")
        
        # Calculate remaining capacity
        max_new_tokens = max(1, model_max_length - input_len)
        logger.info(f"✨ Auto-setting max tokens to: {max_new_tokens} (Context: {model_max_length} - Prompt: {input_len})")

    # Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Use the high limit we calculated
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id  # Ensure model stops when "done"
        )

    # Decode and print ONLY the new tokens (stripping the prompt)
    generated_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    
    print("\n" + "="*40)
    print(f"📝 Prompt: {prompt}")
    print("-" * 40)
    print(f"🤖 Response:\n{generated_text}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple local model loader and text generator.")
    parser.add_argument("model_id", type=str, help="Path to local model or HF model ID")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=-1, help="Max new tokens to generate (-1 for auto)")

    args = parser.parse_args()

    generate_text(args.model_id, args.prompt, args.max_tokens)
