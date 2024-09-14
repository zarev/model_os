"""Main FastAPI application module"""

import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import httpx

from utils import LoadRequest, PromptRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages lifespan of app"""
    # Application startup
    logger.info("Initializing application...")
    yield
    # Application shutdown
    logger.info("Shutting down application...")

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

PHI_SERVICE_URL = os.getenv("PHI_SERVICE_URL", "http://phi_service:8000")
WHISPER_SERVICE_URL = os.getenv("WHISPER_SERVICE_URL", "http://whisper_service:8001")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the index.html file"""
    with open("static/index.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.post("/load")
async def load(request: LoadRequest) -> dict[str, str]:
    """Loads specified models by sending requests to model services."""
    try:
        async with httpx.AsyncClient() as client:
            phi_response = await client.post(f"{PHI_SERVICE_URL}/load")
            whisper_response = await client.post(f"{WHISPER_SERVICE_URL}/load")
        
        if phi_response.status_code == 200 and whisper_response.status_code == 200:
            return {"status": "200 OK", "message": "Models loaded successfully"}
        else:
            return {"status": "500 Internal Server Error", "message": "Failed to load models"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

@app.post("/prompt")
async def prompt(request: PromptRequest):
    """Processes a text prompt and optional image and audio by sending requests to model services."""
    try:
        async with httpx.AsyncClient() as client:
            if request.audio:
                whisper_response = await client.post(f"{WHISPER_SERVICE_URL}/transcribe", json={"audio": request.audio})
                if whisper_response.status_code == 200:
                    request.text += " " + whisper_response.json()["transcription"]
            
            phi_response = await client.post(f"{PHI_SERVICE_URL}/generate", json={
                "text": request.text,
                "image": request.image
            })
        
        if phi_response.status_code == 200:
            return {"response": phi_response.json()["response"]}
        else:
            raise HTTPException(status_code=phi_response.status_code, detail="Error processing prompt")
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing prompt: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
