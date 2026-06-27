import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes.audio import router as audio_router
from .api.routes.history import router as history_router
from .config import HEATMAP_DIR, UPLOAD_DIR
from .database import init_db


app = FastAPI(
    title="DeepVoice Detection API",
    description="Web API for DeepVoice audio detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio_router)
app.include_router(history_router)

FRONTEND_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
FRONTEND_DIST = os.path.join(FRONTEND_ROOT, "dist")
FRONTEND_STATIC_DIR = FRONTEND_DIST if os.path.isdir(FRONTEND_DIST) else FRONTEND_ROOT
FRONTEND_INDEX = os.path.join(FRONTEND_STATIC_DIR, "index.html")

app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="static")
app.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health():
    return {"status": "ok", "service": "DeepVoice Detection API"}


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_fallback(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(FRONTEND_INDEX)


@app.on_event("startup")
def on_startup():
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)
