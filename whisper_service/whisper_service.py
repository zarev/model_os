"""WHISPER model service"""
import sys
import os
import logging
import threading
from fastapi import FastAPI, HTTPException

from models.models import load_whisper_model
from config import WHISPER, BASE_MODEL_PATH
from utils import process_audio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
whisper_model = None

def load_model():
    """Function to load the WHISPER model."""
    global whisper_model
    try:
        whisper_model = load_whisper_model()
        logger.info("WHISPER model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading WHISPER model: {str(e)}")

@app.post("/load")
async def load():
    """Loads the WHISPER model."""
    try:
        load_model()
        return {"status": "200 OK", "message": "WHISPER model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading WHISPER model: {str(e)}")

# Load the model in a background thread to avoid blocking startup
threading.Thread(target=load_model).start()

@app.post("/transcribe")
async def transcribe(audio: str):
    """Transcribes audio using the WHISPER model."""
    global model
    if not model:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load first.")
    try:
        audio_samples = process_audio(audio)
        input_features = model.processor(audio_samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = input_features.to(model.model.device)
        pred_ids = model.model.generate(input_features.input_features, max_new_tokens=440)
        transcription = model.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
