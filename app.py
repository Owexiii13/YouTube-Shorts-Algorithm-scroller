from enum import Enum
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from data_logger import DataLogger
from model import ShortsAIModel

app = FastAPI(title="YouTube Shorts AI Personalizer", version="1.0.0")

# Enable CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
model = ShortsAIModel()
data_logger = DataLogger()


class EventRequest(BaseModel):
    video_id: str
    channel_id: str
    title: str = ""
    description: str = ""
    captions: str = ""
    event_type: str
    watched_percent: float = 0.0
    mood: str = "Neutral"


class PredictionRequest(BaseModel):
    video_id: str
    channel_id: str
    title: str = ""
    description: str = ""
    captions: str = ""


class UserAction(str, Enum):
    viewed = "viewed"
    scrolled = "scrolled"
    liked = "liked"
    disliked = "disliked"
    shared = "shared"
    commented = "commented"
    trusted_channel = "trusted_channel"
    blocked_channel = "blocked_channel"


class LogVideoRequest(BaseModel):
    session_id: str = "anonymous"
    video_id: Optional[str] = None
    channel_id: Optional[str] = None
    watch_percentage: float = Field(default=0.0, ge=0.0, le=1.0)
    watch_duration_seconds: float = Field(default=0.0, ge=0.0)
    user_action: UserAction
    metadata: Dict[str, str] = Field(default_factory=dict)
    timestamp: Optional[float] = None


class ChunkSubmissionRequest(BaseModel):
    chunk_id: str
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=1)
    events: List[LogVideoRequest] = Field(default_factory=list)


@app.get("/")
async def root():
    return {"message": "YouTube Shorts AI Personalizer API", "status": "running"}


# Legacy endpoint kept for compatibility.
@app.post("/event")
async def process_event(request: EventRequest):
    try:
        result = model.process_event(
            video_id=request.video_id,
            channel_id=request.channel_id,
            event_type=request.event_type,
            watched_percent=request.watched_percent,
            mood=request.mood,
        )
        return {"status": "success", "corrections_made": result.get("corrections_made", 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoint kept for compatibility.
@app.post("/next")
async def get_prediction(request: PredictionRequest):
    try:
        result = model.predict_score(
            video_id=request.video_id,
            channel_id=request.channel_id,
            title=request.title,
            description=request.description,
            captions=request.captions,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/log_video")
async def log_video(request: LogVideoRequest):
    try:
        return data_logger.log_video(request.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunk_status")
async def chunk_status(chunk_id: Optional[str] = None):
    try:
        return data_logger.get_chunk_status(chunk_id=chunk_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit_chunk")
async def submit_chunk(request: ChunkSubmissionRequest):
    try:
        return data_logger.submit_chunk(request.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/channel_status")
async def get_channel_status(channel_id: str):
    try:
        result = model.get_channel_status(channel_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/buffer_size")
async def get_buffer_size():
    try:
        buffer_size = model.get_buffer_size()
        return {"buffer_size": buffer_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mood")
async def get_current_mood():
    try:
        current_mood = model.get_current_mood()
        return {"current_mood": current_mood}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mood")
async def set_mood(mood: str):
    try:
        model.set_mood(mood)
        return {"status": "success", "current_mood": model.get_current_mood()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mood/suggest")
async def suggest_mood():
    try:
        suggestion = model.suggest_mood_change()
        return {"suggested_mood": suggestion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting YouTube Shorts AI Personalizer backend...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
