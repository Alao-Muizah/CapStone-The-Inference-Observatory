from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import uuid
from datetime import datetime

from services.inference import run_single_strategy, run_all_strategies
from services.attention import extract_attention
from services.embeddings import compute_similarity

# ── Creating the FastAPI app ─────────────────────────────────────────────────

app = FastAPI(
    title='Inference Observatory API',
    description="Local inference engine for all 10 decoding strategies",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunAllRequest(BaseModel):

    prompt: str
    max_tokens: Optional[int] = 150

    beam_size: Optional[int] = 5

    temperature: Optional[float] = 0.7

    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9

    tktp_k: Optional[int] = 50        
    tktp_p: Optional[float] = 0.9     
    ttk_temp: Optional[float] = 0.7   
    ttk_k: Optional[int] = 50         
    ttp_temp: Optional[float] = 0.7   
    ttp_p: Optional[float] = 0.9      
    ttkp_temp: Optional[float] = 0.7  
    ttkp_k: Optional[int] = 50        
    ttkp_p: Optional[float] = 0.9     

class RunStrategyRequest(BaseModel):
    
    strategy: str
    params: RunAllRequest

class AttentionRequest(BaseModel):
    
    word: str     
    prompt: str

class SimilarityRequest(BaseModel):
   
    texts: dict

# ── ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Inference Observatory",
        "model": "Qwen2.5-0.5B"
    }

@app.get("/health")
async def health():
    
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.post("/api/run-strategy")
async def run_strategy_endpoint(request: RunStrategyRequest):
    
    try:
        start_time = time.time()

        output = await run_single_strategy(
            strategy=request.strategy,
            params=request.params
        )

        output["total_time_ms"] = round((time.time() - start_time) * 1000)
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-all")
async def run_all(request: RunAllRequest):
   
    try:
        start_time = time.time()

        outputs = await run_all_strategies(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            beam_size=request.beam_size,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            tktp_k=request.tktp_k,
            tktp_p=request.tktp_p,
            ttk_temp=request.ttk_temp,
            ttk_k=request.ttk_k,
            ttp_temp=request.ttp_temp,
            ttp_p=request.ttp_p,
            ttkp_temp=request.ttkp_temp,
            ttkp_k=request.ttkp_k,
            ttkp_p=request.ttkp_p,
        )

        return {
            "outputs": outputs,
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "total_time_ms": round((time.time() - start_time) * 1000)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/attention")
async def attention(request: AttentionRequest):
  
    try:
        result = await extract_attention(
            prompt=request.prompt,
            word=request.word
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/similarity")
async def similarity(request: SimilarityRequest):
    
    try:
        scores = await compute_similarity(request.texts)
        return {"scores": scores}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── Run the server ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting Inference Observatory backend...")
    print("API docs available at: http://localhost:8000/docs")
    print("Model: Qwen2.5-0.5B (loads on first request)")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=True
    )
