"""
Similarity API Routes
Define API endpoints related to similarity calculation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.similarity_service import SimilarityService

# Create router
router = APIRouter(prefix="/similarity", tags=["similarity"])

class SimRequest(BaseModel):
    """Similarity request model"""
    code: str
    query: str

@router.post("")
async def calculate_similarity(req: SimRequest):
    """Calculate the similarity between code and query"""
    try:
        result = SimilarityService.calculate_similarity(req.code, req.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}") 