import sys
import psutil
import time
import torch
import random
from model import load_model_and_tokenizer, cleanup_model
from evolutionary_sponge import SystemMonitor


def _estimate_safe_target_seq_len(model, device: str, context_limit: int) -> int:
    """Estimate a safe input length based on current free GPU memory.

    For decoder-only models, KV cache grows roughly linearly with sequence
    length. A common approximation per token is:
      2 (K,V) * num_layers * hidden_size * bytes_per_elem
    """
    # Keep headroom for generation buffers and allocator fragmentation.
    SAFETY_BYTES = int(1.5 * 1024**3)
    MIN_TARGET = 512

    requested = max(50, context_limit - 256)
    
    if not isinstance(model, torch.nn.Module):
        return requested

    if device != "cuda" or not torch.cuda.is_available():
        return requested

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        num_layers = getattr(model.config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(model.config, "n_layer", 32)
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", 4096)

        # Most paths here run fp16 on ROCm/CUDA.
        bytes_per_elem = 2
        bytes_per_token = 2 * int(num_layers) * int(hidden_size) * bytes_per_elem
        kv_budget = max(0, int(free_bytes) - SAFETY_BYTES)
        safe_tokens = kv_budget // max(1, bytes_per_token)

        safe_target = int(max(MIN_TARGET, min(requested, safe_tokens)))
        # Leave some room for generation tokens.
        safe_target = min(safe_target, max(MIN_TARGET, context_limit - 256))
        return safe_target
    except Exception:
        # Conservative fallback if memory introspection fails.
        return min(requested, 4096)

def run_context_exhaustion(
    model_id,
    num_requests=5,
    is_quantized=False,
    progress_callback=None,
    context_mode="combined",
    force_decode_tokens=64,
    disable_eos_stop=False,
):
    """
    Sends multiple sequential requests with inputs designed to be just below the context limit.
    This aims to exhaust the available context window capacity and monitor degradation.
    """
    try:
        if progress_callback:
            progress_callback({
                "status": "starting",
                "message": (
                    f"Initializing Context Exhaustion Attack (Requests: {num_requests}, "
                    f"Quantized: {is_quantized}, Mode: {context_mode})..."
                )
            })
        
        quant_mode = is_quantized if isinstance(is_quantized, str) else ("bnb-nf4" if is_quantized else "none")
        tokenizer, model, device, quant_label = load_model_and_tokenizer(model_id, quantize=quant_mode)
        
        # Determine model context max window
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit is None:
            context_limit = getattr(model.config, "n_positions", 1024)
            
        if progress_callback:
            progress_callback({"status": "running", "message": f"Context window limit: {context_limit} tokens"})

        # Target sequence length (VRAM-aware on GPU).
        target_seq_len = _estimate_safe_target_seq_len(model, device, context_limit)
        if progress_callback:
            progress_callback({"status": "running", "message": f"Targeting inputs of length: {target_seq_len} tokens..."})

        results = []
        overall_start = time.time()
        
        for i in range(num_requests):
            req_num = i + 1
            if progress_callback:
                progress_callback({"status": "eval", "message": f"Preparing Request {req_num}/{num_requests}..."})
            
            # Start with current target, then reduce on OOM until it fits.
            effective_target = target_seq_len
            error_msg = None
            generated_text = ""
            prompt_text = ""
            output_tokens = 0
            prefill_duration = 0.0
            decode_duration = 0.0
            monitor = SystemMonitor(device="cuda" if "cuda" in str(device) else "cpu")
            monitor.start()

            for attempt in range(5):
                try:
                    # Generate purely valid ASCII text to prevent detokenization expansion & invalid unicode errors
                    # Generate more than enough chars, then truncate to the exact token limit
                    base_text = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", k=effective_target * 5))
                    input_ids = tokenizer(base_text, truncation=True, max_length=effective_target).input_ids.to(device)
                    
                    # Pad if for some reason it's shorter than expected
                    if input_ids.shape[1] < effective_target:
                        padding = torch.randint(50, 100, (1, effective_target - input_ids.shape[1])).to(device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                        
                    prompt_text = tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)

                    # Phase A: prefill-only forward pass (context pressure source)
                    prefill_start = time.perf_counter()
                    with torch.no_grad():
                        prefill_out = model(
                            input_ids=input_ids,
                            use_cache=True,
                            return_dict=True,
                        )
                    if device == "cuda":
                        torch.cuda.synchronize()
                    prefill_duration = time.perf_counter() - prefill_start

                    if context_mode == "prefill_only":
                        output_tokens = 0
                        generated_text = "[prefill_only]"
                    else:
                        # Phase B: controlled decode stress (fixed token budget)
                        # Exhaust the rest of the context window
                        # Adding safety buffer of 16 tokens for GGUF/llama.cpp detokenization discrepancies
                        safety_buffer = 1 if isinstance(model, torch.nn.Module) else 16
                        max_new_tokens = max(1, context_limit - effective_target - safety_buffer)
                        if force_decode_tokens > max_new_tokens:
                             max_new_tokens = force_decode_tokens
                        min_new_tokens = max_new_tokens
                        eos_token_id = None if disable_eos_stop else tokenizer.eos_token_id

                        decode_start = time.perf_counter()
                        with torch.no_grad():
                            out = model.generate(
                                input_ids,
                                min_new_tokens=min_new_tokens,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                use_cache=True,
                                eos_token_id=eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        if device == "cuda":
                            torch.cuda.synchronize()
                        decode_duration = time.perf_counter() - decode_start

                        output_tokens = out.shape[1] - input_ids.shape[1]
                        generated_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

                    del input_ids
                    del prefill_out
                    error_msg = None
                    break
                except Exception as e:
                    msg = str(e)
                    is_oom = "out of memory" in msg.lower() or "hip out of memory" in msg.lower() or "llama_decode returned 1" in msg.lower() or "context" in msg.lower()
                    if not is_oom or effective_target <= 512 or attempt == 4:
                        error_msg = msg
                        generated_text = f"Error: {error_msg}"
                        break

                    # Reduce target and retry.
                    effective_target = max(512, int(effective_target * 0.75))
                    if progress_callback:
                        progress_callback({
                            "status": "eval",
                            "message": (
                                f"  OOM at ~{target_seq_len} tokens; retrying with {effective_target} tokens..."
                            ),
                        })
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            monitor.stop(token_count=output_tokens)

            score, max_temp, tps, cpu, gpu, duration, avg_power, energy = monitor.get_score()
            
            latency_per_input_tok_ms = (duration * 1000.0 / max(1, effective_target))
            energy_per_input_tok_mj = (energy * 1000.0 / max(1, effective_target))

            msg = (
                f"Req {req_num} complete ({duration:.2f}s) | "
                f"Prefill: {prefill_duration:.2f}s | Decode: {decode_duration:.2f}s | "
                f"in_tok: {effective_target} out_tok: {output_tokens} | "
                f"lat/tok: {latency_per_input_tok_ms:.3f} ms | "
                f"CPU: {cpu:.1f}%"
            )
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
                "effective_input_tokens": effective_target,
                "output_tokens": output_tokens,
                "prefill_duration": prefill_duration,
                "decode_duration": decode_duration,
                "latency_per_input_token_ms": latency_per_input_tok_ms,
                "energy_per_input_token_mj": energy_per_input_tok_mj,
                "output": generated_text,
                "prompt_ending": prompt_text,
                "error": error_msg
            })

            # For context exhaustion, we might want memory pressure to remain or stack 
            # In a real environment, KV cache builds up from previous concurrent requests, 
            # here we verify performance during high context generation. 

        overall_duration = time.time() - overall_start
        cleanup_model(model, tokenizer)
        model = None
        tokenizer = None
        import gc
        gc.collect()
        
        # Calculate summary/score equivalent to best_result for frontend compat
        valid_results = [r for r in results if not r.get("error")]
        best_duration = overall_duration
        best_cpu = sum([r["avg_cpu"] for r in valid_results]) / max(1, len(valid_results)) if valid_results else 0
        best_gpu = sum([r["avg_gpu"] for r in valid_results]) / max(1, len(valid_results)) if valid_results else 0
        total_energy = sum([r.get("energy_joules", 0) for r in results])
        avg_prefill_duration = (
            sum(r.get("prefill_duration", 0.0) for r in results if not r.get("error")) /
            max(1, len([r for r in results if not r.get("error")]))
        )
        avg_decode_duration = (
            sum(r.get("decode_duration", 0.0) for r in results if not r.get("error")) /
            max(1, len([r for r in results if not r.get("error")]))
        )
        avg_latency_per_token_ms = (
            sum(r.get("latency_per_input_token_ms", 0.0) for r in results if not r.get("error")) /
            max(1, len([r for r in results if not r.get("error")]))
        )
        avg_energy_per_token_mj = (
            sum(r.get("energy_per_input_token_mj", 0.0) for r in results if not r.get("error")) /
            max(1, len([r for r in results if not r.get("error")]))
        )
        
        # Get the output from the longest running request (the "best" exhaustion)
        best_req = max(results, key=lambda x: x["duration"], default=results[0] if results else {})
        
        final_result = {
            "score": total_energy, # Use energy as score proxy
            "duration": overall_duration,
            "avg_cpu": best_cpu,
            "avg_gpu": best_gpu,
            "energy_joules": total_energy,
            "prefill_duration": best_req.get("prefill_duration", 0.0),
            "decode_duration": best_req.get("decode_duration", 0.0),
            "avg_prefill_duration": avg_prefill_duration,
            "avg_decode_duration": avg_decode_duration,
            "input_tokens": best_req.get("effective_input_tokens", target_seq_len),
            "output_tokens": sum(r.get("output_tokens", 0) for r in results if not r.get("error")),
            "latency_per_input_token_ms": best_req.get("latency_per_input_token_ms", 0.0),
            "energy_per_input_token_mj": best_req.get("energy_per_input_token_mj", 0.0),
            "avg_latency_per_input_token_ms": avg_latency_per_token_ms,
            "avg_energy_per_input_token_mj": avg_energy_per_token_mj,
            "context_mode": context_mode,
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
