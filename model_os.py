"""model_os module"""

# uvicorn model_os:app --reload --host 0.0.0.0 --port 8000
# pylint: disable=fixme

import os
from os import path
from io import BytesIO
import queue
import threading
from contextlib import asynccontextmanager
from typing import Optional
import base64
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from PIL import Image
from pydub import AudioSegment
from torch import float16
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForSpeechSeq2Seq
from numpy import float32, int16, array, frombuffer
from pydub import AudioSegment
import httpx

from config import *
from utils import PromptRequest, LoadRequest
from models.models import load_phi_model
# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

ROOT_DIR = "."
HOME_PATH = path.expanduser("~")

PHI_SERVICE_URL = os.getenv("PHI_SERVICE_URL", "Endpoint not found or not defined")
WHISPER_SERVICE_URL = os.getenv("WHISPER_SERVICE_URL", "Endpoint not found or not defined")

@asynccontextmanager
async def lifespan(my_app: FastAPI):  # pylint: disable=unused-argument
    """Manages lifespan of app"""
    # Application startup
    yield

app = FastAPI(lifespan=lifespan)

model_enpoint = "generate"

@app.post("/process")
async def process_request(request: Request):
    try:
        # Determine if the request contains an image or text
        content_type = request.headers.get("Content-Type")
        
        if "multipart/form-data" in content_type:
            form = await request.form()
            if "image" in form:
                # Forward to phi_service
                files = {"image": form["image"].file}
                async with httpx.AsyncClient() as client:
                    response = await client.post(f"{PHI_SERVICE_URL}/{model_enpoint}", files=files)
                return response.json()
            elif "text" in form:
                # Forward to phi_service for text processing
                data = {"text": form["text"]}
                async with httpx.AsyncClient() as client:
		    # todo: define endpoint from config
                    response = await client.post(f"{PHI_SERVICE_URL}/{model_enpoint}", json=data)
                return response.json()
        elif "application/json" in content_type:
            data = await request.json()
            if "image" in data or "text" in data:
                # Forward to phi_service with image URL
                async with httpx.AsyncClient() as client:
                    response = await client.post(f"{PHI_SERVICE_URL}/{model_enpoint}", json=data)
                return response.json()
        else:
            raise HTTPException(status_code=400, detail="Unsupported Content-Type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
