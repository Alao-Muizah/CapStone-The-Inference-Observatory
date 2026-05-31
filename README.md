#  The Inference Observatory

A local inference engine that runs **10 decoding strategies simultaneously** on a real language model, visualizing how each one works at the token level. Built as a capstone project to demonstrate deep understanding of transformer decoding mechanisms.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square&logo=fastapi)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![HTML](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-orange?style=flat-square&logo=html5)

---

## What This Is

Most decoding-strategy comparisons rely on cloud APIs, where techniques such as greedy decoding, beam search, top-k sampling, and top-p sampling are often approximated or hidden behind proprietary systems. As a result, you cannot directly observe how these strategies influence token selection.

This project runs everything locally using the Qwen2.5-0.5B-Instruct model, providing direct access to the generation process and model internals.

Every decoding strategy executes a real model.generate() call with its corresponding parameters. Attention visualizations are derived from actual forward passes, and semantic similarity metrics are computed using real sentence embeddings. No outputs are simulated, or approximated.

---

## Why This Matters

When interacting with commercial LLM APIs such as GPT or Claude, you typically provide a prompt and receive generated text. The underlying decoding process—the algorithm responsible for selecting each token—remains hidden.

This project exposes that process, allowing you to observe, compare, and measure how different decoding strategies affect generation behavior.

---

## Decoding Strategies

The app implements 10 strategies organized into 4 groups based on their mechanism:

### Group 1 — Deterministic Decoding
These strategies make no random choices. The same prompt always produces the same output.

| Strategy | Mechanism |
|---|---|
| **Greedy** | At every step, picks the single token with the highest probability. Fast but can get stuck in repetitive loops. |
| **Beam Search** | Maintains N candidate sequences simultaneously. At each step expands all beams and keeps the top N by total log-probability. Slower than greedy by a factor of N but finds globally better sequences. |

### Group 2 — Pure Sampling
These strategies sample from the probability distribution. The same prompt produces different outputs every run.

| Strategy | Mechanism |
|---|---|
| **Random Sampling** | Samples from the raw unmodified distribution (temperature=1.0, no filtering). Every token in the vocabulary has a nonzero chance of being picked. |
| **Temperature** | Divides all logits by a temperature value before softmax. Low temperature sharpens the distribution (more focused). High temperature flattens it (more random). |

### Group 3A — Basic Constrained Sampling
These strategies filter the distribution before sampling, preventing very low probability tokens from being selected.

| Strategy | Mechanism |
|---|---|
| **Top-k** | Keeps only the top k most probable tokens. All others are set to zero and the remaining k are renormalized. Prevents absurd low-probability choices. |
| **Top-p / Nucleus** | Keeps the smallest set of tokens whose cumulative probability exceeds p. Dynamic nucleus — adapts to model confidence at each step. When confident, the nucleus is small. When uncertain, it grows. |

### Group 3B — Hybrid Constrained Sampling
These strategies combine multiple constraints simultaneously.

| Strategy | Mechanism |
|---|---|
| **Top-k + Top-p** | Applies both k and p filters. Top-k first, then top-p on the remaining tokens. More conservative than either alone. |
| **Temp + Top-k** | Temperature reshapes the distribution first, then top-k filters the reshaped distribution. The order matters — temperature changes which tokens top-k keeps. |
| **Temp + Top-p** | Temperature reshapes the distribution, then nucleus sampling filters from the reshaped distribution. |
| **Temp + Top-k + Top-p** | All three constraints applied together. Most constrained combination — temperature reshapes, top-k truncates, top-p refines. |

---

## Why Qwen2.5-0.5B-Instruct

Model choice was deliberate. Here is the reasoning:

**Why not a cloud API (Claude, GPT-4, Groq)?**

Cloud APIs are black boxes. They accept prompts and return text. You cannot access the logits, control the decoding algorithm at the token level, or extract attention weights. If you call Groq and request beam search, Groq approximates it or ignores it — you have no way to verify the mechanism is actually running. For a project about mechanism correctness, cloud APIs are fundamentally unsuitable.

**Why not GPT-2?**

GPT-2 was the obvious small local model choice. It runs on CPU and is well understood. However GPT-2's output quality is noticeably poor — its text is often incoherent and repetitive. When comparing decoding strategies side by side, poor base quality makes it harder to see what the strategy is actually doing versus what the model's limitations are causing. A student looking at GPT-2 outputs might conclude beam search is bad when really GPT-2 is just weak.

**Why Qwen2.5-0.5B-Instruct specifically?**

- **Size** — 0.5 billion parameters, approximately 988MB on disk. Runs comfortably on CPU with 8GB RAM.
- **Quality** — Modern architecture (2024). Coherent, grammatically correct outputs even at 150 tokens. The strategy differences are visible in the output, not buried under model noise.
- **Real decoding access** — HuggingFace Transformers gives full control over `model.generate()` parameters. Every strategy uses the actual documented parameters: `do_sample`, `num_beams`, `top_k`, `top_p`, `temperature`.
- **Attention extraction** — `output_attentions=True` returns the full attention tensor from all 24 layers and 14 heads. Real self-attention data, not approximated.
- **License** — Apache 2.0. Fully open for academic and portfolio use.

The tradeoff is download size (~1GB) and inference speed (10-60 seconds per strategy on CPU). This is acceptable for a demonstration tool where correctness matters more than speed.

---

## Features

### 10 Decoding Strategies
Each strategy runs `model.generate()` with its specific parameters. Results are displayed with token count, latency in milliseconds, and tokens per second.

### Token Replay Visualizer
After each strategy completes, its output tokens are replayed one by one with a flash animation. This visualizes the autoregressive nature of generation — each token built on all previous tokens. Local inference generates all tokens first then replays them sequentially.

### Self-Attention Heatmap
After running analysis, the app runs a forward pass through the model with `output_attentions=True`. It returns:
- **Layer 0, Head 0** — raw attention weights for the first layer and first head
- **Averaged** — attention normalized and averaged across all 24 layers and 14 heads

The heatmap shows how much each token attended to every other token during processing.

### Consistency Tracker
Runs any selected strategy 5 times on the same prompt. Deterministic strategies (greedy, beam search) produce identical outputs every run. Sampling strategies produce varied outputs. The tracker makes temperature's effect on variance measurable rather than theoretical.

### Inference Tradeoff Dashboard
A summary table showing latency, token count, and tokens per second for all 10 strategies. Directly shows the speed cost of beam search versus greedy versus sampling strategies.

### Semantic Similarity
Embeds each strategy's output using `all-MiniLM-L6-v2` (sentence-transformers) and computes pairwise cosine similarity. Shows that greedy and beam search are often semantically close even when the exact words differ. Shows that high-temperature sampling diverges semantically from deterministic strategies.

Within-group similarity updates after each group completes. Cross-group similarity runs as part of the analysis pass.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| **Local Model** | Qwen2.5-0.5B-Instruct | Real decoding control, attention access, CPU-compatible |
| **ML Framework** | PyTorch + HuggingFace Transformers | Industry standard, full model internals access |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Fast, accurate semantic similarity, 22MB |
| **Backend** | Python + FastAPI | Async, automatic validation, excellent docs at /docs |
| **Frontend** | HTML + CSS + JavaScript | No build tools, zero dependencies, runs directly in browser |
| **Tokenizer (live counter)** | tiktoken cl100k_base | Approximate real-time token counting without model call |

---

## Project Structure

---

## Local Setup

### Prerequisites
- Python 3.10 or higher
- 8GB RAM minimum
- ~2GB free disk space (model cache)
- Internet connection for first run (model download)

### Backend

```bash
# Navigate to backend
cd inference-observatory/backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
echo "# No API keys needed — fully local" > .env

# Start the server
python main.py
```

On first run, Qwen2.5-0.5B-Instruct downloads automatically (~988MB). This takes a few minutes depending on your connection. Subsequent runs load from cache in 20-40 seconds.

The API will be available at `http://localhost:8000`.
Interactive API docs at `http://localhost:8000/docs`.

### Frontend

No build step needed. Open `frontend/index.html` directly in Chrome or Firefox.

Make sure the backend is running before clicking Run All Strategies.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `POST` | `/api/run-strategy` | Run one strategy by name |
| `POST` | `/api/run-all` | Run all strategies (used by consistency tracker) |
| `POST` | `/api/attention` | Extract self-attention matrix |
| `POST` | `/api/similarity` | Compute cosine similarity between output texts |

### Example: Run a single strategy

```bash
curl -X POST http://localhost:8000/api/run-strategy \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "beam_search",
    "params": {
      "prompt": "Explain how neural networks learn",
      "max_tokens": 150,
      "beam_size": 5
    }
  }'
```

Response:
```json
{
  "strategy": "beam_search",
  "text": "Neural networks learn by adjusting their weights...",
  "tokens": ["Neural", " networks", " learn", ...],
  "token_count": 47,
  "latency_ms": 8420.5,
  "prompt_tokens": 8,
  "finish_reason": "stop"
}
```

---

## How Each Component Teaches

| Feature | Concept Demonstrated |
|---|---|
| Live token counter | Tokenization — text splits into subword units, not words |
| Side-by-side outputs | Decoding strategy effect on output style and content |
| Token replay | Autoregressive generation — one token at a time |
| Attention heatmap | Self-attention — every token relating to every other token |
| Consistency tracker | Temperature and variance — determinism vs randomness |
| Tradeoff dashboard | Speed/quality tradeoff — beam search cost vs greedy |
| Semantic similarity | Embedding space — meaning preserved across different words |

---

## Performance on CPU

Approximate inference times on a standard laptop CPU (8GB RAM):

| Strategy | ~Time |
|---|---|
| Greedy | 15-30s |
| Beam Search (beams=5) | 60-120s |
| Random Sampling | 15-30s |
| Temperature | 15-30s |
| Top-k | 15-30s |
| Top-p | 15-30s |
| Hybrid strategies | 15-30s |

Beam search is the slowest because it runs beam_size forward passes at each token step. All times are for 150 output tokens. First run is slower due to model loading.

---

## Known Limitations

- **CPU only** — No GPU acceleration. Inference is slow by production standards. This is intentional — the project prioritizes mechanism correctness over speed.
- **Single user** — FastAPI runs in reload mode for development. Not production-hardened.
- **Model size** — ~988MB download on first run. Requires stable internet connection.
- **Tiktoken mismatch** — The live token counter uses tiktoken (cl100k_base encoding) while the model uses Qwen's own tokenizer. The live count is approximate. Exact counts appear per-strategy after inference.

---

## Background

This project was built as a capstone assignment exploring transformer inference mechanisms. The goal was to go beyond theoretical understanding — to build something that makes abstract concepts like beam search and nucleus sampling observable and measurable.

The design decision to use local inference rather than cloud APIs was central. It meant slower execution but genuine mechanism correctness. Every number in the tradeoff dashboard, every cell in the attention heatmap, every similarity score came from the model's actual computations — not approximations.

---

## Author

Built by Alao Muizah.
