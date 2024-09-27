"""PHI model service"""

import logging
import threading
from fastapi import FastAPI, HTTPException
from models.models import load_phi_model
#from config import PHI, BASE_MODEL_PATH
from utils import PromptRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
#phi_model = load_model()

#@app.on_event("startup")
#def load_model():
#    """Function to load the PHI model during application startup."""
#    global phi_model
#    try:
#        phi_model = load_phi_model()
#        logger.info("PHI model loaded successfully during startup.")
#    except Exception as e:
#        logger.error(f"Error loading PHI model: {str(e)}")
        # Optionally, you might want to exit the application if the model fails to load
#        import sys
#        sys.exit(1)


def load_model():
    """Function to load the PHI model."""
    global phi_model
    try:
        phi_model = load_phi_model() #DB1
        logger.info("PHI model loaded successfully inside load_model().")
        return phi_model
    except Exception as e:
        logger.error(f"Error loading PHI model: {str(e)}") #DB1 Incorrect path_or_model_id

phi_model = load_model()

#@app.post("/load")
#async def load():
#    """Loads the PHI model. This is obsolete and no longer relevant."""
#    try:
#        load_model()
#        return {"status": "200 OK", "message": "PHI model loaded successfully inside load() endpoint"}
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=f"Error loading PHI model: {str(e)}")

# Load the model in a background thread to avoid blocking startup
#threading.Thread(target=load_model).start()

#print('service: is phi present?', phi_model)
#print('service: is processor present?', phi_model.model.processor)


@app.post("/generate")
async def generate(request: PromptRequest):
    """Generates a response using the PHI model."""
    global phi_model
    if not phi_model:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load first.")
    try:
        response = phi_model.generate(image=request.image, text=request.text, audio=None)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
