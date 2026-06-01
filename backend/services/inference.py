import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Global Model Variables ─────────────────────────────────────────────────

tokenizer = None
model = None


# ── Model Loading ──────────────────────────────────────────────────────────

def load_model():
   
    global tokenizer, model

    if model is not None:
        return

    print(f"Loading {MODEL_NAME}...")
    print("First load takes 20-40 seconds. Cached after that.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)

    model.eval()
    print("Model loaded successfully.")


# ── Tokenization ───────────────────────────────────────────────────────────

def tokenize_prompt(prompt: str) -> dict:
    
    return tokenizer(prompt, return_tensors="pt")


# ── Decoding ───────────────────────────────────────────────────────────────

def decode_output(output_ids, prompt_length: int):
   
    generated_ids = output_ids[0][prompt_length:]

    tokens = [
        tokenizer.decode([token_id], skip_special_tokens=True)
        for token_id in generated_ids
    ]

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return full_text, tokens


# ── Shared Result Builder ──────────────────────────────────────────────────

def build_result(strategy: str, output_ids, inputs: dict, start: float, max_tokens: int) -> dict:
    
    prompt_length = inputs['input_ids'].shape[1]
    full_text, tokens = decode_output(output_ids, prompt_length)

    return {
        "strategy":     strategy,
        "text":         full_text,
        "tokens":       tokens,
        "token_count":  len(tokens),
        "latency_ms":   round((time.time() - start) * 1000, 2),
        "prompt_tokens": prompt_length,
        "finish_reason": "length" if len(tokens) >= max_tokens else "stop",
    }


# ── Strategy Functions ─────────────────────────────────────────────────────

def run_greedy(inputs, max_tokens: int) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('greedy', output_ids, inputs, start, max_tokens)


def run_beam_search(inputs, max_tokens: int, beam_size: int) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=beam_size,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('beam_search', output_ids, inputs, start, max_tokens)


def run_sampling(inputs, max_tokens: int) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('sampling', output_ids, inputs, start, max_tokens)


def run_temperature(inputs, max_tokens: int, temperature: float) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('temperature', output_ids, inputs, start, max_tokens)


def run_top_k(inputs, max_tokens: int, top_k: int) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('top_k', output_ids, inputs, start, max_tokens)


def run_top_p(inputs, max_tokens: int, top_p: float) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('top_p', output_ids, inputs, start, max_tokens)


def run_top_k_top_p(inputs, max_tokens: int, top_k: int, top_p: float) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('top_k_top_p', output_ids, inputs, start, max_tokens)


def run_temp_top_k(inputs, max_tokens: int, temperature: float, top_k: int) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('temp_top_k', output_ids, inputs, start, max_tokens)


def run_temp_top_p(inputs, max_tokens: int, temperature: float, top_p: float) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('temp_top_p', output_ids, inputs, start, max_tokens)


def run_temp_top_k_top_p(inputs, max_tokens: int, temperature: float, top_k: int, top_p: float) -> dict:
    
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return build_result('temp_top_k_top_p', output_ids, inputs, start, max_tokens)


# ── run_single_strategy ────────────────────────────────────────────────────

async def run_single_strategy(strategy: str, params) -> dict:
   
    load_model()
    inputs = tokenize_prompt(params.prompt)
    max_tokens = params.max_tokens

    strategy_map = {
        'greedy':           lambda: run_greedy(inputs, max_tokens),
        'beam_search':      lambda: run_beam_search(inputs, max_tokens, params.beam_size),
        'sampling':         lambda: run_sampling(inputs, max_tokens),
        'temperature':      lambda: run_temperature(inputs, max_tokens, params.temperature),
        'top_k':            lambda: run_top_k(inputs, max_tokens, params.top_k),
        'top_p':            lambda: run_top_p(inputs, max_tokens, params.top_p),
        'top_k_top_p':      lambda: run_top_k_top_p(inputs, max_tokens, params.tktp_k, params.tktp_p),
        'temp_top_k':       lambda: run_temp_top_k(inputs, max_tokens, params.ttk_temp, params.ttk_k),
        'temp_top_p':       lambda: run_temp_top_p(inputs, max_tokens, params.ttp_temp, params.ttp_p),
        'temp_top_k_top_p': lambda: run_temp_top_k_top_p(inputs, max_tokens, params.ttkp_temp, params.ttkp_k, params.ttkp_p),
    }

    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}")

    return strategy_map[strategy]()


# ── run_all_strategies ─────────────────────────────────────────────────────

async def run_all_strategies(
    prompt: str,
    max_tokens: int = 150,
    beam_size: int = 5,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 0.7,
    tktp_k: int = 50,
    tktp_p: float = 0.9,
    ttk_temp: float = 0.7,
    ttk_k: int = 50,
    ttp_temp: float = 0.7,
    ttp_p: float = 0.9,
    ttkp_temp: float = 0.7,
    ttkp_k: int = 50,
    ttkp_p: float = 0.9,
) -> list[dict]:
    
    load_model()
    inputs = tokenize_prompt(prompt)
    results = []

    strategy_calls = [
        ('greedy',           lambda: run_greedy(inputs, max_tokens)),
        ('beam_search',      lambda: run_beam_search(inputs, max_tokens, beam_size)),
        ('sampling',         lambda: run_sampling(inputs, max_tokens)),
        ('temperature',      lambda: run_temperature(inputs, max_tokens, temperature)),
        ('top_k',            lambda: run_top_k(inputs, max_tokens, top_k)),
        ('top_p',            lambda: run_top_p(inputs, max_tokens, top_p)),
        ('top_k_top_p',      lambda: run_top_k_top_p(inputs, max_tokens, tktp_k, tktp_p)),
        ('temp_top_k',       lambda: run_temp_top_k(inputs, max_tokens, ttk_temp, ttk_k)),
        ('temp_top_p',       lambda: run_temp_top_p(inputs, max_tokens, ttp_temp, ttp_p)),
        ('temp_top_k_top_p', lambda: run_temp_top_k_top_p(inputs, max_tokens, ttkp_temp, ttkp_k, ttkp_p)),
    ]

    for strategy_name, strategy_fn in strategy_calls:
        try:
            results.append(strategy_fn())
        except Exception as e:
            results.append({"strategy": strategy_name, "error": str(e)})

    return results