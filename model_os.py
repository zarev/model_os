"""model_os module"""

# uvicorn model_os:app --reload
# pylint: disable=fixme

from io import BytesIO
import queue
import threading
from contextlib import asynccontextmanager
from typing import Optional
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from pydub import AudioSegment
from torch import float16
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np


# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

@asynccontextmanager
async def lifespan(my_app: FastAPI):  # pylint: disable=unused-argument
    """Manages lifespan of app"""
    # Application startup
    # load phi model
    load(LoadRequest(num_gpus=1, models_per_gpu=1, model_name="phi"))
    # load(LoadRequest(num_gpus=1, models_per_gpu=1, model_name="whisper"))
    yield
    # Application shutdown
    models.clear()

app = FastAPI(lifespan=lifespan)

from pydub import AudioSegment

class Model:
    """
    A class representing an inference model.

    Attributes:
    -----------
    model_path : str
        The file path to the model.
    index : ints
        The index identifying the model instance.
    loaded : bool
        A flag indicating whether the model is currently loaded.

    Methods:
    --------
    load():
        Loads the model and initializes it.

    run(input: tuple[Optional[str], str], output: queue.Queue) -> None:
        Processes input text and image using the loaded model and generates a response.

    prompt(image: Optional[str], text: str) -> str:
        Runs the `run` method in a separate thread and retrieves the result.
    """

    def __init__(self, model_path: str, index: int):
        """
        Initializes the Model class with a given model path and index.

        Parameters:

        model_path : str
            The file path to the model.
        index : int
            The index identifying the model instance.
        """
        self.model = None
        self.model_path = model_path

        print('init model path\n', self.model_path)

        self.is_whisper = False
        self.is_whisper = model_path.count('whisper') == 1

        self.model_type = AutoModelForCausalLM
        print('init model type\n', self.model_type)
        self.use_safetensors = True if self.is_whisper else False

        self.index = index
        self.loaded = False
        self.processor = None
        self.tokenizer_stream = None

    def load(self):
        """Loads model components"""
        # tensorflow model loading logic
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=self.model_path,
            # use_safetensors=self.use_safetensors,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            # Use _attn_implementation='eager' to disable flash attention, or 'flash_attention_2'
            _attn_implementation="eager")

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.loaded = True

    def run(self, input_data: tuple[Optional[str], Optional[str], str], output: queue.Queue) -> None:
        """
        Processes input text and image using the loaded model and generates a response.

        Parameters:
        -----------
        input : tuple
            A tuple containing the input image path (or None) and the input text.
        output : queue.Queue
            A queue used to store the output or error message for retrieval by the calling thread.

        Notes:
        ------
        - If an image is provided, it is loaded and included in the prompt.
        - The method handles exceptions during input processing and places
          error messages into the output queue.
        - The generated response is incrementally built by decoding tokens from the model output.
        - The response is placed into the output queue when processing is complete.
        """
        # TODO add file input option
        input_image, input_audio, input_text = input_data

        if input_audio:
            # model_path = "C:\\models\\whisper"
            model_path = "/app/models/whisper"

            whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path, torch_dtype="auto", low_cpu_mem_usage=True, use_safetensors=True, device_map="cuda", trust_remote_code=True
            )

            whisper_processor = AutoProcessor.from_pretrained(model_path)

            print('server receieved audio')
            print('server audio string chars\n', input_audio[:20])
            # Assuming input_audio is a base64-encoded string
            b64_bytes = base64.b64decode(input_audio)
            print('server audio bytes\n', b64_bytes[:20])
            print('len audio bytes\n', len(b64_bytes)) # check
            arr_contents = np.frombuffer(b64_bytes, dtype=np.float32)
            audio_array_int16 = (arr_contents * 32768).astype(np.int16)

            print('audio_array_int16 int16\n', audio_array_int16[:20])

            audio_segment = AudioSegment(data=audio_array_int16.tobytes(), sample_width=2, frame_rate=44100, channels=1)

            audio_segment = audio_segment.set_frame_rate(16000)

            # export audio to check if it survived the journey
            audio_segment.export("/app/decode_test.wav", format="wav")
            
            # Convert AudioSegment to numpy array
            audio_samples = np.array(audio_segment.get_array_of_samples())

            input_features = whisper_processor(audio_samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            input_features = input_features.to("cuda", dtype=float16)

            attention_mask = input_features['attention_mask']
            # Model inference with attention mask
            pred_ids = whisper.generate(input_features.input_features, attention_mask=attention_mask, max_new_tokens=200)
            # Decode the predicted IDs to get the transcription
            pred_text = whisper_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            
            print(f"transciption:\n{pred_text}")
            # Append the transcription to the input text
            input_text += pred_text
    
        messages = [{
            "role": "user",
            "content": ''}]

        image_obj = None
        # If there is image load it and add tag to prompt
        if input_image:
            image_data = base64.b64decode(input_image)
            image_obj = Image.open(BytesIO(image_data))
            messages[0]["content"] += "<|image_1|>"

        messages[0]["content"] += f"{input_text}"

        templated_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)

        try:
            inputs = self.processor(templated_prompt, image_obj, return_tensors="pt").to("cuda:0")

        except RuntimeError as e:
            # for some reason the error only goes to output
            # if there is a print statement first, otherwise
            # output is empty on client.
            # TODO investigate why this is
            print(flush=True)
            output.put(f"Error occurred during input processing: {e}")
            return

        generation_args = {
		"max_new_tokens": 2048,
		"temperature": 0.1,
		"do_sample": True,}

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args)

        # Remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        output.put(response)
        output.put(None)

    def generate(self, image: Optional[str], audio: Optional[str], text: str) -> str:
        """
        Runs the `run` method in a separate thread and retrieves the result.

        Parameters:
        image (str): Path to the image, or None if not provided.
        text (str): The text prompt for processing.

        Returns:
        str: The processing result or an error message.
        """
        output = queue.Queue()
        thread = threading.Thread(
            target=self.run, args=((image, audio, text), output))
        thread.daemon = True
        thread.start()

        result = []
        while thread.is_alive():
            try:
                text = output.get()
                if text is None:
                    output.put(
                        'Error: Empty response')
                    break
                result.append(text)
            except queue.Empty:
                output.put('''Error: No response received from the thread.
                           The processing might be taking longer than expected.
                           Consider checking the model, inputs, and system resources.''')
                break

        thread.join()
        return "".join(result)


# Store loaded models in a dictionary
models = {}


class LoadRequest(BaseModel):
    '''Initial load request on start with default values'''
    num_gpus: int = 1
    models_per_gpu: int = 1
    model_name: str = "phi"


@app.post("/load")
def load(request: LoadRequest):
    """
    Loads a model into memory if it's not already loaded.

    This function checks if a model specified by the request is already loaded.
    If not, it loads the model into memory and makes it available for processing.
    """

    root_path = "/app/models/"
    if request.model_name not in models:
        for gpu_index in range(request.num_gpus):
            for _ in range(request.models_per_gpu):
                model_path = "/app/models/phi"
                model = Model(model_path, gpu_index)
                model.load()
                models[request.model_name] = model

    return {"status": "200 OK", "model_name": request.model_name}


class PromptRequest(BaseModel):
    '''Definition of the client request'''
    model: int = 0
    image: Optional[str] = None
    audio: Optional[str] = None
    text: str
    model_name: str = "phi"


@app.post("/prompt")
def prompt(request: PromptRequest):
    """
    Processes a text prompt (and optional image) using a specified model.

    This function retrieves the specified model, processes the input text and image,
    and returns the model's generated response.
    """

    if request.model_name not in models:
        raise HTTPException(
            status_code=404, detail="Model not found but tried to prompt.")

    model = models[request.model_name]

    request_prompt = request.text

    audio = None
    if request.audio is not None:
        audio = request.audio

    image = None
    if request.image is not None:
        image = request.image

    response_text = model.generate(image=image, text=request_prompt, audio=audio)

    return {"response": response_text}
