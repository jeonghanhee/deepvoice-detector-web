from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from ...database import get_db, AudioAnalysis
from ...schemas import HistoryResponse, AnalysisResult, StatsResponse

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    total = db.query(func.count(AudioAnalysis.id)).scalar()
    items = (
        db.query(AudioAnalysis)
        .order_by(AudioAnalysis.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return HistoryResponse(
        total=total,
        items=[AnalysisResult.model_validate(r) for r in items],
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(AudioAnalysis.id)).scalar() or 0
    fake_count = db.query(func.count(AudioAnalysis.id)).filter(AudioAnalysis.is_fake == 1).scalar() or 0
    real_count = total - fake_count
    avg_conf = db.query(func.avg(AudioAnalysis.confidence)).scalar() or 0.0
    avg_proc = db.query(func.avg(AudioAnalysis.processing_time)).scalar() or 0.0

    return StatsResponse(
        total_analyses=total,
        fake_count=fake_count,
        real_count=real_count,
        avg_confidence=round(float(avg_conf), 2),
        avg_processing_time=round(float(avg_proc), 3),
    )


@router.delete("/{analysis_id}")
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    record = db.query(AudioAnalysis).filter(AudioAnalysis.id == analysis_id).first()
    if not record:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(record)
    db.commit()
    return {"success": True, "message": "삭제되었습니다."}
