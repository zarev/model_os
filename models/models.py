"""model_os module"""

# models/models.py

import os
#from config import PHI, WHISPER, BASE_MODEL_PATH
from models.model import Model

PHI = os.getenv("PHI", "phi path not found")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "phi path not found")

print("PHI", PHI)
print("BASEMODELPATH", BASE_MODEL_PATH)

def load_phi_model():
    phi_base_name = PHI.rsplit('/', maxsplit=1)[-1]
    phi_path = f"/app/model_tensors/Phi-3-vision-128k-instruct/"
    phi_model = Model(phi_path, 0) #DB1 end: path invalid
    phi_model.load()
    print('model loading inside load_phi_model', phi_model)
    return phi_model

#def load_whisper_model():
#    whisper_base_name = WHISPER.rsplit('/', maxsplit=1)[-1]
#    whisper_path = f"{BASE_MODEL_PATH}/{whisper_base_name}/"
#    whisper_model = Model(whisper_path, 0)
#    whisper_model.load()
#    return whisper_model
