"""model_os module"""

# models/models.py

from config import PHI, WHISPER, BASE_MODEL_PATH
from models.model import Model

def load_phi_model():
    phi_base_name = PHI.rsplit('/', maxsplit=1)[-1]
    phi_path = f"{BASE_MODEL_PATH}/{phi_base_name}/"
    phi_model = Model(phi_path, 0)
    phi_model.load()
    return phi_model

def load_whisper_model():
    whisper_base_name = WHISPER.rsplit('/', maxsplit=1)[-1]
    whisper_path = f"{BASE_MODEL_PATH}/{whisper_base_name}/"
    whisper_model = Model(whisper_path, 0)
    whisper_model.load()
    return whisper_model
