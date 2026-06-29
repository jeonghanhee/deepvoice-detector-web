import asyncio
import os
from datetime import datetime

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ...config import UPLOAD_DIR
from ...database import AnalysisLog, AudioAnalysis, get_db
from ...schemas import AnalysisResponse, AnalysisResult
from ...services.detector import analyze_audio_sync


router = APIRouter(prefix="/api/audio", tags=["audio"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024


def _log(db: Session, analysis_id: int, level: str, message: str) -> None:
    log = AnalysisLog(analysis_id=analysis_id, log_level=level, message=message)
    db.add(log)
    db.commit()


@router.post("/upload", response_model=AnalysisResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"{timestamp}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    content = await file.read()
    file_size = len(content)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds 50MB.")

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    record = AudioAnalysis(
        filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        is_fake=0,
        confidence=0.0,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    _log(db, record.id, "INFO", f"File uploaded: {file.filename} ({file_size} bytes)")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_audio_sync, file_path)
    except Exception as exc:
        _log(db, record.id, "ERROR", f"Analysis failed: {exc}")
        if os.path.exists(file_path):
            os.remove(file_path)
        db.delete(record)
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed: {exc}",
        ) from exc

    record.is_fake = result["is_fake"]
    record.confidence = result["confidence"]
    record.processing_time = result["processing_time"]
    record.duration = result["duration"]
    record.sample_rate = result["sample_rate"]
    record.spectral_centroid = result["spectral_centroid"]
    record.zero_crossing_rate = result["zero_crossing_rate"]
    db.commit()
    db.refresh(record)

    label = "AI-generated voice" if result["is_fake"] else "real voice"
    decision_model = result.get("decision_model", "tts")
    detail_type = result.get("detail_type", "-")
    _log(
        db,
        record.id,
        "INFO",
        f"Analysis complete: {label} "
        f"(confidence {result['confidence']}%, model {decision_model}, "
        f"detail {detail_type}, "
        f"processing {result['processing_time']}s)",
    )

    return AnalysisResponse(
        success=True,
        message=f"Analysis complete: detected as {label}.",
        result=AnalysisResult.model_validate(record),
        frequency_data=result["freq_data"],
        decision_model=decision_model,
        final_label=result.get("final_label"),
        detail_type=detail_type,
        model_results=result.get("model_results"),
        heatmap_url=result["heatmap_url"],
        heatmap_metadata=result["heatmap_metadata"],
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    record = db.query(AudioAnalysis).filter(AudioAnalysis.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis result not found.")
    return AnalysisResponse(
        success=True,
        message="Analysis result loaded.",
        result=AnalysisResult.model_validate(record),
    )
