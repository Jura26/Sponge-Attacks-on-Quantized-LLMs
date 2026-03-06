import sys
import psutil
import time
import torch
import random
from model import load_model_and_tokenizer, cleanup_model
from sponge_attack import SystemMonitor

def run_context_exhaustion(model_id, num_requests=5, is_quantized=False, progress_callback=None):
    """
    Sends multiple sequential requests with inputs designed to be just below the context limit.
    This aims to exhaust the available context window capacity and monitor degradation.
    """
    try:
        if progress_callback:
            progress_callback({"status": "starting", "message": f"Initializing Context Exhaustion Attack (Requests: {num_requests}, Quantized: {is_quantized})..."})
        
        tokenizer, model, device, quant_label = load_model_and_tokenizer(model_id, quantize=is_quantized)
        
        # Determine model context max window
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit is None:
            context_limit = getattr(model.config, "n_positions", 1024)
            
        if progress_callback:
            progress_callback({"status": "running", "message": f"Context window limit: {context_limit} tokens"})

        # Target sequence length
        # E.g., just below the limit to allow for a few generated output tokens without crashing outright
        target_seq_len = max(50, context_limit - 100) 
        if progress_callback:
            progress_callback({"status": "running", "message": f"Targeting inputs of length: {target_seq_len} tokens..."})

        results = []
        overall_start = time.time()
        
        for i in range(num_requests):
            req_num = i + 1
            if progress_callback:
                progress_callback({"status": "eval", "message": f"Preparing Request {req_num}/{num_requests}..."})
            
            # Create a random proxy token tensor to avoid having real semantics which might finish early
            vocab_size = getattr(tokenizer, 'vocab_size', 50257) 
            input_ids = torch.randint(100, vocab_size - 100, (1, target_seq_len)).to(device)
            
            # Start monitoring
            monitor = SystemMonitor(device="cuda" if "cuda" in str(device) else "cpu")
            monitor.start()
            
            error_msg = None
            generated_text = ""
            prompt_text = ""
            try:
                # Ask the model to generate a few tokens
                out = model.generate(
                    input_ids,
                    max_new_tokens=context_limit - target_seq_len, # Generate until we hit the context limit
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                output_tokens = out.shape[1] - input_ids.shape[1]
                prompt_text = tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True) # decode ending to not overwhelm UI
                generated_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                monitor.stop(token_count=output_tokens)
            except Exception as e:
                error_msg = str(e)
                generated_text = f"Error: {error_msg}"
                monitor.stop(token_count=0)

            score, max_temp, tps, cpu, gpu, duration, avg_power, energy = monitor.get_score()
            
            msg = f"Req {req_num} complete ({duration:.2f}s) | CPU: {cpu:.1f}% | GPU: {gpu:.1f}%"
            if error_msg:
                msg = f"Req {req_num} FAILED: {error_msg} | ({duration:.2f}s) | CPU: {cpu:.1f}%"
                
            if progress_callback:
                progress_callback({
                    "status": "eval", "message": msg
                })

            results.append({
                "request": req_num,
                "duration": duration,
                "avg_cpu": cpu,
                "avg_gpu": gpu,
                "energy_joules": energy,
                "output": generated_text,
                "prompt_ending": prompt_text,
                "error": error_msg
            })

            # For context exhaustion, we might want memory pressure to remain or stack 
            # In a real environment, KV cache builds up from previous concurrent requests, 
            # here we verify performance during high context generation. 

        overall_duration = time.time() - overall_start
        cleanup_model(model, tokenizer)
        
        # Calculate summary/score equivalent to best_result for frontend compat
        best_duration = max([r["duration"] for r in results if not r.get("error")] + [0])
        best_cpu = max([r["avg_cpu"] for r in results if not r.get("error")] + [0])
        best_gpu = max([r["avg_gpu"] for r in results if not r.get("error")] + [0])
        total_energy = sum([r.get("energy_joules", 0) for r in results])
        
        # Get the output from the longest running request (the "best" exhaustion)
        best_req = max(results, key=lambda x: x["duration"], default=results[0] if results else {})
        
        final_result = {
            "score": best_duration, # Use max latency as a score proxy
            "duration": overall_duration,
            "avg_cpu": best_cpu,
            "avg_gpu": best_gpu,
            "energy_joules": total_energy,
            "input_tokens": target_seq_len,
            "output_tokens": sum(r.get("output_tokens", 50) for r in results if not r.get("error")),
            "prompt": f"Random context sequence size {target_seq_len}\nEnded with: ...{best_req.get('prompt_ending', '')}",
            "output": best_req.get("output", "Empty output...")
        }

        if progress_callback:
            progress_callback({
                "status": "complete",
                "message": "Context exhaustion attack completed.",
                "result": final_result
            })
            
        return final_result

    except Exception as e:
        if progress_callback:
            progress_callback({"status": "error", "message": f"Fatal string error: {str(e)}"})
        return None
