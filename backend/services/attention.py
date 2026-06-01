import torch
import services.inference as inference_module


async def extract_attention(prompt: str, word: str = None) -> dict:
    
    inference_module.load_model()

    inputs = inference_module.tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"][0]
    raw_tokens = inference_module.tokenizer.convert_ids_to_tokens(
        input_ids.tolist()
    )

    clean_tokens = []
    for token in raw_tokens:
        cleaned = token.replace("Ġ", " ").replace("▁", " ").strip()
        if cleaned == "":
            cleaned = token
        clean_tokens.append(cleaned)

    with torch.no_grad():
        outputs = inference_module.model(
            **inputs,
            output_attentions=True,
            attn_implementation="eager",
        )

    attentions = outputs.attentions
   
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    # ── View 1: Layer 0, Head 0 ───────────────────────────────────────
  
    layer0_head0 = attentions[0][0, 0].detach().tolist()

    layer0_head0 = [
        [round(v, 4) for v in row]
        for row in layer0_head0
    ]

    stacked = torch.stack(attentions)

    averaged = stacked.mean(dim=0).mean(dim=1)

    avg_matrix = averaged[0].detach().tolist()

    normalized_avg = []
    for row in avg_matrix:
        row_max = max(row)
        if row_max > 0:
            normalized_avg.append([round(v / row_max, 4) for v in row])
        else:
            normalized_avg.append([round(v, 4) for v in row])

    return {
        "tokens": clean_tokens,
        "num_tokens": len(clean_tokens),
        "num_layers": num_layers,
        "num_heads": num_heads,

       
        "layer0_head0": layer0_head0,

        
        "attention_matrix": normalized_avg,
    }