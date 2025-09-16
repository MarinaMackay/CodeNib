"""
Similarity API Routes
Define API endpoints related to similarity calculation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..services.similarity_service import SimilarityService

# Create router
router = APIRouter(prefix="/similarity", tags=["similarity"])

class SimRequest(BaseModel):
    """Similarity request model"""
    code: str
    query: str

class ConfigRequest(BaseModel):
    """Configuration request model"""
    model: Optional[str] = None
    device: Optional[str] = None

@router.post("")
async def calculate_similarity(req: SimRequest):
    """Calculate the similarity between code and query"""
    try:
        result = SimilarityService.calculate_similarity(req.code, req.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}")

@router.post("/configure")
async def configure_service(req: ConfigRequest):
    """Configure model and device settings"""
    try:
        if req.model is not None:
            SimilarityService.set_model(req.model)
        if req.device is not None:
            SimilarityService.set_device(req.device)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get information about available models"""
    try:
        return {
            "available_models": SimilarityService.get_available_models(),
            "current_config": SimilarityService.get_current_config()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

@router.get("/config")
async def get_current_config():
    """Get current configuration"""
    try:
        return SimilarityService.get_current_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}") 