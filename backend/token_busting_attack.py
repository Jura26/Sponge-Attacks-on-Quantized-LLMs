import time
from model import load_model_and_tokenizer, cleanup_model
from evolutionary_sponge import SystemMonitor

def generate_bpe_nightmare_prompt(length: int = 500) -> str:
    """
    Generates a prompt designed to break BPE/tokenizers by using
    alternating obscure unicode characters, emoji combinations,
    and interspersed invisible characters to force maximum token count.
    """
    # Combining emojis, zero-width joiners, and obscure scripts
    nightmare_chars = [
        "👩‍👩‍👦‍👦", "\u200D", "_\u200B", "a\u0328", "\U000e0061\U000e0062",
        "أ", "﷽", "𒈙", "𒐫", " ﷺ", ":_:", "><", "```", "'''", "[[[", "]]]",
        "🧑🏽‍🚀", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🏴‍☠️", "🤦🏿‍♂️", "🧠", "🧬", "🪐",
        "🇺🇳", "🇺🇸", "🇯🇵", "🇫🇷", "🇩🇪", "🇬🇧", "🇺🇦",
        "e\u0301", "o\u0308", "n\u0303", "a\u030a", "u\u0308", "i\u0307",
        "\u200C", "\u200E", "\u200F", "\u2060", "\u2061", "\u2062", "\u2063",
        "\uFE0E", "\uFE0F", "\u034F", "\u00AD", "\u00A0",
        "अ", "आ", "इ", "उ", "ए", "क", "ष",
        "語", "漢", "語", "字", "かな", "カナ",
        "😊", "😂", "🤖", "💥", "✨", "🔥", "⚡",
        "=!=", "====", "::::", "////", "\\\\", "----", "____",
    ]
    prompt = ""
    for i in range(length):
        prompt += nightmare_chars[i % len(nightmare_chars)]
    return prompt

def run_token_busting_attack(
    model_id: str,
    num_requests: int = 5,
    is_quantized: str = None,
    progress_callback=None
):
    def update(msg: str):
        if progress_callback:
            progress_callback({"status": "running", "log": msg})
        else:
            print(msg)

    def update_eval(msg: str):
        if progress_callback:
            progress_callback({"status": "eval", "message": msg})
        else:
            print(msg)

    update(f"Starting Token-Busting Attack on {model_id} (Quant: {is_quantized})...")
    
    # Load model
    tokenizer, model, device, quant_label = load_model_and_tokenizer(model_id, is_quantized)
    
    if not model:
        update("Error: Failed to load model. Attack aborted.")
        return

    update(f"Loaded {model_id} successfully.")

    monitor = SystemMonitor(device="cuda" if "cuda" in str(device) else "cpu")
    monitor.start()
    
    total_latency = 0
    total_output_tokens = 0
    total_input_tokens = 0
    total_input_chars = 0
    last_prompt = ""
    last_output_text = ""
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(model.config, "n_positions", 4096)

    try:
        for i in range(num_requests):
            update(f"--- Request {i+1}/{num_requests} ---")
            update_eval(f"Token-Busting: request {i+1}/{num_requests} starting")
            
            prompt = generate_bpe_nightmare_prompt(600 + i*100)
            last_prompt = prompt

            input_batch = tokenizer(prompt, return_tensors="pt")
            input_ids = input_batch.input_ids
            prompt_token_count = input_ids.shape[1]
            update(f"Formatted {len(prompt)} characters. Tokenizer exploded this into {prompt_token_count} tokens!")
            update_eval(
                f"Token-Busting: request {i+1} input tokens={prompt_token_count}, chars={len(last_prompt)}"
            )

            # Respect context window: keep room for generation tokens.
            max_prompt_tokens = max(1, int(context_limit) - 64)
            if prompt_token_count > max_prompt_tokens:
                input_ids = input_ids[:, :max_prompt_tokens]
                prompt_token_count = input_ids.shape[1]
                last_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                update(
                    f"Prompt truncated to {prompt_token_count} tokens to fit context window ({context_limit})."
                )

            total_input_chars += len(last_prompt)
            
            req_start = time.time()
            
            # Force generation
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            req_end = time.time()
            req_latency = req_end - req_start
            
            output_tokens = output.shape[1] - input_ids.shape[1]
            last_output_text = tokenizer.decode(
                output[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            
            total_latency += req_latency
            total_input_tokens += prompt_token_count
            total_output_tokens += output_tokens
            
            update(f"Generated {output_tokens} tokens in {req_latency:.2f}s")
            update_eval(
                f"Token-Busting: request {i+1} output tokens={output_tokens}, latency={req_latency:.2f}s"
            )
            
            time.sleep(0.5)

    except Exception as e:
        update(f"Error during Token-Busting attack: {str(e)}")
    
    finally:
        cleanup_model(model, tokenizer)
        monitor.stop(token_count=total_output_tokens)

    # Calculate hardware stats
    score, _max_temp, _tps, cpu_avg, gpu_avg, duration, power_avg, energy_joules = monitor.get_score()

    result = {
        "score": score,
        "duration": duration,
        "avg_cpu": cpu_avg,
        "avg_gpu": gpu_avg,
        "avg_power": power_avg,
        "energy_joules": energy_joules,
        "input_tokens": total_input_tokens,
        "input_chars": total_input_chars,
        "output_tokens": total_output_tokens,
        "prompt": last_prompt,
        "output": last_output_text
    }

    if progress_callback:
        progress_callback({
            "status": "complete",
            "log": "Token-Busting Complete.",
            "result": result
        })

    return result