"""PHI model service"""

import logging
from fastapi import FastAPI, HTTPException
from models import Model
from config import PHI, BASE_MODEL_PATH
from utils import PromptRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

@app.post("/load")
async def load():
    """Loads the PHI model."""
    global model
    try:
        model_path = f"{BASE_MODEL_PATH}/{PHI.split('/')[-1]}"
        model = Model(model_path, 0)
        model.load()
        return {"status": "200 OK", "message": "PHI model loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading PHI model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading PHI model: {str(e)}")

@app.post("/generate")
async def generate(request: PromptRequest):
    """Generates a response using the PHI model."""
    global model
    if not model:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load first.")
    try:
        response = model.generate(image=request.image, text=request.text, audio=None)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
