from typing import List, Optional
import webbrowser

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

from data_logger import DataLogger
from model import ShortsAIModel

GITHUB_ISSUE_URL = "https://github.com/Owexiii13/YouTube-Shorts-Algorithm-scroller/issues/new?template=data-contribution.md"

app = FastAPI(title="YouTube Shorts AI Personalizer", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ShortsAIModel()
logger = DataLogger(".")


class EventRequest(BaseModel):
    video_id: str
    channel_id: str = "unknown"
    title: str = ""
    description: str = ""
    captions: str = ""
    tags: List[str] = Field(default_factory=list)
    duration_seconds: int = Field(default=0, ge=0)
    event_type: str
    watched_percent: float = 0.0
    mood: str = "Neutral"


class PredictionRequest(BaseModel):
    video_id: str
    channel_id: str = "unknown"
    title: str = ""
    description: str = ""
    captions: str = ""
    tags: List[str] = Field(default_factory=list)
    duration_seconds: int = Field(default=0, ge=0)
    mood: str = "Neutral"


class LogVideoRequest(BaseModel):
    watch_percentage: float = Field(default=0.0, ge=0.0, le=1.0)
    user_action: str = "neutral"
    algorithm_action: str = "none"
    reason: str = ""

    @field_validator("user_action")
    @classmethod
    def validate_user_action(cls, value: str) -> str:
        allowed = {"liked", "disliked", "neutral"}
        if value not in allowed:
            raise ValueError(f"user_action must be one of: {sorted(allowed)}")
        return value

    @field_validator("algorithm_action")
    @classmethod
    def validate_algorithm_action(cls, value: str) -> str:
        allowed = {"liked", "disliked", "scrolled", "none"}
        if value not in allowed:
            raise ValueError(f"algorithm_action must be one of: {sorted(allowed)}")
        return value


class SubmitChunkRequest(BaseModel):
    chunk_file: str


@app.get("/")
async def root():
    return {
        "message": "YouTube Shorts AI Personalizer API",
        "status": "running",
        "shared_model_records": model.base_model.get("record_count", 0),
    }


@app.post("/event")
async def process_event(request: EventRequest):
    try:
        result = model.process_event(
            video_id=request.video_id,
            channel_id=request.channel_id,
            event_type=request.event_type,
            watched_percent=request.watched_percent,
            mood=request.mood,
            title=request.title,
            description=request.description,
            captions=request.captions,
            tags=request.tags,
            duration_seconds=request.duration_seconds,
        )
        return {"status": "success", "corrections_made": result.get("corrections_made", 0)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/next")
async def get_prediction(request: PredictionRequest):
    try:
        return model.predict_action(
            video_id=request.video_id,
            channel_id=request.channel_id,
            title=request.title,
            description=request.description,
            captions=request.captions,
            tags=request.tags,
            duration_seconds=request.duration_seconds,
            mood=request.mood,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/log_video")
async def log_video(request: LogVideoRequest):
    try:
        result = logger.log_video(request.model_dump())
        model.export_contribution_model(logger.export_path(), logger.watch_count)
        return {
            "status": "success",
            "chunk_file": result.chunk_file,
            "chunk_count": result.chunk_count,
            "completed_chunk": result.completed_chunk,
            "privacy_notice": logger.chunk_status().get("privacy_notice"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chunk_status")
async def chunk_status():
    try:
        return logger.chunk_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/submit_chunk")
async def submit_chunk(request: SubmitChunkRequest):
    try:
        mark_result = logger.mark_uploaded(request.chunk_file)
        webbrowser.open(GITHUB_ISSUE_URL)
        return {"status": "success", **mark_result, "issue_url": GITHUB_ISSUE_URL}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Contribution model file not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/channel_status")
async def get_channel_status(channel_id: str):
    try:
        return model.get_channel_status(channel_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/buffer_size")
async def get_buffer_size():
    try:
        return {"buffer_size": model.get_buffer_size()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/mood")
async def get_current_mood():
    try:
        return {"current_mood": model.get_current_mood()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/mood")
async def set_mood(mood: str):
    try:
        model.set_mood(mood)
        return {"status": "success", "current_mood": model.get_current_mood()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/mood/suggest")
async def suggest_mood():
    try:
        return {"suggested_mood": model.suggest_mood_change()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    print("Starting YouTube Shorts AI Personalizer backend...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
