import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Suppress transformers progress bars and verbose logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_id: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"🔄 Loading tokenizer for {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"✅ Tokenizer loaded.")

        logger.info(f"🔄 Loading model {model_id} on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cuda":
            logger.info(f"✅ Model loaded on CUDA.")
        else:
            # If not using device_map="auto" (CPU), explicitly move to device not really needed for CPU but good practice
            model.to(device)
            logger.info(f"✅ Model loaded on CPU.")
            
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

def generate_text(model_id: str, prompt: str, max_new_tokens: int = -1):
    """
    Loads model and generates text based on prompt.
    """
    tokenizer, model, device = load_model_and_tokenizer(model_id)

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
