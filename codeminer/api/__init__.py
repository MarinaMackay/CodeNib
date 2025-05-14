"""
CodeMiner API Package
Provides API server framework and similarity calculation functionality
"""

from .base import BaseAPI
from .services.similarity_service import SimilarityService as SimilarityAPI

# Auto-start on import (if in main module)
BaseAPI.auto_start_on_import()

__all__ = ["BaseAPI", "SimilarityAPI"] 