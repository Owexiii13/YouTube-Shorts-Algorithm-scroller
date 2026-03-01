import json
import os
import time
from typing import Any, Dict, List, Optional


class DataLogger:
    def __init__(self, data_file: str = "shorts_event_log.json"):
        self.data_file = data_file
        self.events: List[Dict[str, Any]] = []
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.load_data()

    def load_data(self) -> None:
        if not os.path.exists(self.data_file):
            self.save_data()
            return

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.events = data.get("events", [])
            self.chunks = data.get("chunks", {})
        except Exception:
            self.events = []
            self.chunks = {}
            self.save_data()

    def save_data(self) -> None:
        data = {
            "events": self.events,
            "chunks": self.chunks,
        }
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def log_video(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "event_id": f"evt_{int(time.time() * 1000)}_{len(self.events)}",
            "session_id": payload.get("session_id", "anonymous"),
            "video_id": payload.get("video_id") or None,
            "channel_id": payload.get("channel_id") or None,
            "watch_percentage": payload.get("watch_percentage", 0.0),
            "watch_duration_seconds": payload.get("watch_duration_seconds", 0.0),
            "user_action": payload.get("user_action"),
            "metadata": payload.get("metadata", {}),
            "timestamp": payload.get("timestamp", time.time()),
            "received_at": time.time(),
        }
        self.events.append(event)
        self.save_data()
        return {
            "status": "logged",
            "event_id": event["event_id"],
            "total_events": len(self.events),
        }

    def submit_chunk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        chunk_id = payload["chunk_id"]
        chunk_index = payload["chunk_index"]
        total_chunks = payload["total_chunks"]

        chunk_state = self.chunks.get(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "total_chunks": total_chunks,
                "received_chunks": [],
                "last_updated": None,
            },
        )

        if chunk_index not in chunk_state["received_chunks"]:
            chunk_state["received_chunks"].append(chunk_index)
            chunk_state["received_chunks"].sort()

        chunk_state["total_chunks"] = total_chunks
        chunk_state["last_updated"] = time.time()

        events = payload.get("events", [])
        for event in events:
            self.log_video(event)

        self.chunks[chunk_id] = chunk_state
        self.save_data()

        is_complete = len(chunk_state["received_chunks"]) >= total_chunks
        return {
            "status": "complete" if is_complete else "partial",
            "chunk_id": chunk_id,
            "received_chunks": chunk_state["received_chunks"],
            "total_chunks": total_chunks,
            "logged_events": len(events),
        }

    def get_chunk_status(self, chunk_id: Optional[str] = None) -> Dict[str, Any]:
        if chunk_id:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                return {
                    "chunk_id": chunk_id,
                    "status": "not_found",
                    "received_chunks": [],
                    "total_chunks": 0,
                }

            return {
                "chunk_id": chunk_id,
                "received_chunks": chunk["received_chunks"],
                "total_chunks": chunk["total_chunks"],
                "is_complete": len(chunk["received_chunks"]) >= chunk["total_chunks"],
                "last_updated": chunk.get("last_updated"),
            }

        completed = 0
        for chunk in self.chunks.values():
            if len(chunk.get("received_chunks", [])) >= chunk.get("total_chunks", 0):
                completed += 1

        return {
            "total_chunks": len(self.chunks),
            "completed_chunks": completed,
            "pending_chunks": len(self.chunks) - completed,
            "total_logged_events": len(self.events),
        }
