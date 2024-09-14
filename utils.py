"""Utility functions and classes"""

from typing import Optional
import base64
import numpy as np
from pydantic import BaseModel
from pydub import AudioSegment

class LoadRequest(BaseModel):
    """Initial load request on start with default values"""
    num_gpus: int = 1
    models_per_gpu: int = 1
    model_name: str = "phi"

class PromptRequest(BaseModel):
    """Definition of the client request"""
    model: int = 0
    image: Optional[str] = None
    audio: Optional[str] = None
    text: str
    model_name: str = "phi"

def process_audio(input_audio: str) -> np.array:
    """
    Decodes a base64-encoded audio string, converts it to an audio segment,
    resamples it to 16 kHz, and returns the audio samples as a NumPy array.
    """
    b64_bytes = base64.b64decode(input_audio)
    arr_contents = np.frombuffer(b64_bytes, dtype=np.float32)
    audio_array_int16 = (arr_contents * 32768).astype(np.int16)
    audio_segment = AudioSegment(data=audio_array_int16.tobytes(), sample_width=2, frame_rate=44100, channels=1)
    audio_segment = audio_segment.set_frame_rate(16000)
    
    audio_samples = np.array(audio_segment.get_array_of_samples())
    return audio_samples
