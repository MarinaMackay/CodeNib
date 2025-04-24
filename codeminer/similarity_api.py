from functools import lru_cache
from typing import Tuple, AsyncGenerator
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
from contextlib import asynccontextmanager

MAX_SEQ_LENGTH = 8192 # Can be reduced to increase the speed

# Load the model only on the first call
@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
        device="cpu"  # Force CPU usage
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model

def code_text_similarity(code: str, query: str) -> float:
    model = _load_model()
    vec_q, vec_c = model.encode(
        [query, code],
        normalize_embeddings=True, # L2 normalization
        show_progress_bar=False
    )
    score = float(util.cos_sim(vec_q, vec_c).item())
    return score

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Warm model
    _ = _load_model()
    yield

app = FastAPI(
    title="Code–Query Similarity API",
    description="Calculate cosine similarity between code and query using Jina embeddings.",
    version="0.1",
    lifespan=lifespan
)

class SimRequest(BaseModel):
    code: str
    query: str

@app.post("/similarity")
def similarity(req: SimRequest):
    try:
        start = time.time()
        score = code_text_similarity(req.code, req.query)
        latency = round(time.time() - start, 4)
        return {"score": score, "latency_sec": latency}
    except Exception as e:
        return {"error": str(e)}