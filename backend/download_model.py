import argparse
import os
import sys
# Force offline mode off just in case
os.environ["HF_HUB_OFFLINE"] = "0"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Transformers not installed. Run: pip install transformers torch")
    sys.exit(1)

def download_model(model_id):
    print(f"⬇️  Starting download for: {model_id}")
    print("   (This may take a while for large models like gpt2-medium/large)")
    print("-" * 50)
    
    try:
        print("1. Downloading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("✅ Tokenizer OK.")
        
        print(f"2. Downloading Model ({model_id})...")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        print("✅ Model Download Complete!")
        
        print("-" * 50)
        print(f"🎉 {model_id} is now cached locally.")
        print("   You can now restart your backend server and run the attack.")
        
    except Exception as e:
        print("\n❌ Download Failed!")
        print(f"Error: {e}")
        print("\nTip: If this is a rate limit error, try setting a HuggingFace Token:")
        print("     $env:HF_TOKEN='your_token_here'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-medium", help="Model ID to download (e.g., gpt2, gpt2-medium, gpt2-large)")
    args = parser.parse_args()
    
    download_model(args.model)
