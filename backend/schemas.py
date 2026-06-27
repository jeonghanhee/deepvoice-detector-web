from pydantic import BaseModel
from datetime import datetime
from typing import Any, Dict, Optional, List


class AnalysisResult(BaseModel):
    id: int
    filename: str
    file_size: Optional[int]
    duration: Optional[float]
    sample_rate: Optional[int]
    is_fake: int
    confidence: float
    processing_time: Optional[float]
    spectral_centroid: Optional[float]
    zero_crossing_rate: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    result: Optional[AnalysisResult] = None
    frequency_data: Optional[List[float]] = None
    heatmap_url: Optional[str] = None
    heatmap_metadata: Optional[Dict[str, Any]] = None


class HistoryResponse(BaseModel):
    total: int
    items: List[AnalysisResult]


class StatsResponse(BaseModel):
    total_analyses: int
    fake_count: int
    real_count: int
    avg_confidence: float
    avg_processing_time: float
