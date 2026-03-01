from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Set
from pathlib import Path
import re
import webbrowser
import uvicorn
from model import ShortsAIModel

app = FastAPI(title="YouTube Shorts AI Personalizer", version="1.0.0")

GITHUB_ISSUE_URL = "https://github.com/your-org/your-repo/issues/new"
TRAINING_FILE_PATTERN = re.compile(r"^DataForTraining(\d+)\.json$")
session_prompt_state: Dict[str, Set[str]] = {}

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


class SubmitChunkRequest(BaseModel):
    filename: str
    session_id: str = "default"


def get_pending_chunks() -> list[str]:
    pending_files = []
    for file_path in Path(".").glob("DataForTraining*.json"):
        match = TRAINING_FILE_PATTERN.match(file_path.name)
        if match:
            pending_files.append((int(match.group(1)), file_path.name))

    pending_files.sort(key=lambda item: item[0])
    return [filename for _, filename in pending_files]

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


@app.get("/chunk_status")
async def get_chunk_status(session_id: str = "default"):
    try:
        pending_chunks = get_pending_chunks()
        next_chunk = pending_chunks[0] if pending_chunks else None
        prompted_chunks = session_prompt_state.get(session_id, set())
        prompted_this_session = bool(next_chunk and next_chunk in prompted_chunks)

        return {
            "status": "success",
            "pending_chunks": pending_chunks,
            "next_chunk": next_chunk,
            "prompted_this_session": prompted_this_session,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "chunk_status_failed", "message": str(e)})


@app.post("/submit_chunk")
async def submit_chunk(request: SubmitChunkRequest):
    filename = request.filename.strip()
    if not TRAINING_FILE_PATTERN.match(filename):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_filename",
                "message": "Filename must match DataForTrainingN.json format.",
                "filename": filename,
            },
        )

    source_path = Path(filename)
    uploaded_name = filename.replace(".json", "Uploaded.json")
    destination_path = Path(uploaded_name)

    if not source_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "missing_chunk_file",
                "message": "The chunk file does not exist.",
                "filename": filename,
            },
        )

    if destination_path.exists():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "rename_conflict",
                "message": "Uploaded target file already exists.",
                "filename": filename,
                "target": uploaded_name,
            },
        )

    try:
        source_path.rename(destination_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "rename_failed",
                "message": str(e),
                "filename": filename,
                "target": uploaded_name,
            },
        )

    issue_opened = False
    try:
        issue_opened = webbrowser.open(GITHUB_ISSUE_URL, new=2)
    except Exception:
        issue_opened = False

    prompted_chunks = session_prompt_state.setdefault(request.session_id, set())
    prompted_chunks.add(filename)

    return {
        "status": "success",
        "message": "Chunk submitted successfully.",
        "filename": filename,
        "renamed_to": uploaded_name,
        "issue_url": GITHUB_ISSUE_URL,
        "issue_opened": issue_opened,
    }

if __name__ == "__main__":
    print("Starting YouTube Shorts AI Personalizer backend...")
    print("Server will be available at: http://localhost:8000" )
    print("API documentation at: http://localhost:8000/docs" )
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
