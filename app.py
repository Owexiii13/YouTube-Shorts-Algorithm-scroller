from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
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

@app.get("/")
async def root():
    return {"message": "YouTube Shorts AI Personalizer API", "status": "running"}

@app.post("/event")
async def process_event(request: EventRequest):
    try:
        result = model.process_event(
            video_id=request.video_id,
            channel_id=request.channel_id,
            event_type=request.event_type,
            watched_percent=request.watched_percent,
            mood=request.mood
        )
        return {"status": "success", "corrections_made": result.get("corrections_made", 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/next")
async def get_prediction(request: PredictionRequest):
    try:
        result = model.predict_score(
            video_id=request.video_id,
            channel_id=request.channel_id,
            title=request.title,
            description=request.description,
            captions=request.captions
        )
        return result
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
    print("Server will be available at: http://localhost:8000" )
    print("API documentation at: http://localhost:8000/docs" )
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
