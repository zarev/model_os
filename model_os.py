"""model_os module"""

# uvicorn model_os:app --reload
# pylint: disable=no-member
# pylint: disable=fixme

import queue
import threading
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime_genai as og


@asynccontextmanager
async def lifespan(my_app: FastAPI):  # pylint: disable=unused-argument
    """Manages lifespan of app"""
    # Application startup
    load(LoadRequest(num_gpus=1, models_per_gpu=1))
    yield
    # Application shutdown
    models.clear()

app = FastAPI(lifespan=lifespan)


class Model:
    """
    A class representing a model loaded using ONNX.

    Attributes:
    -----------
    model_path : str
        The file path to the ONNX model.
    index : ints
        The index identifying the model instance.
    loaded : bool
        A flag indicating whether the model is currently loaded.

    Methods:
    --------
    load():
        Loads the ONNX model and initializes the processor and tokenizer stream.

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
            The file path to the ONNX model.
        index : int
            The index identifying the model instance.
        """
        self.model = None
        self.model_path = model_path
        self.index = index
        self.loaded = False
        self.processor = None
        self.tokenizer_stream = None

    def load(self):
        """Loads model components"""
        # ONNX specific model loading logic
        self.model = og.Model(self.model_path)  # pylint: disable=no-member
        self.processor = self.model.create_multimodal_processor()
        self.tokenizer_stream = self.processor.create_stream()
        self.loaded = True

    def run(self, input_data: tuple[Optional[str], str], output: queue.Queue) -> None:
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
        input_image, input_text = input_data
        template_prompt = "<|user|>\n"

        image_obj = None
        # If there is image load it and add tag to prompt
        if input_image:
            image_obj = og.Images.open(input_image)  # pylint: disable=no-member
            template_prompt += "<|image_1|>\n"

        template_prompt += f"{input_text}\n<|end|>\n<|assistant|>"

        try:
            inputs = self.processor(template_prompt, images=image_obj)
        except RuntimeError as e:
            # for some reason the error only goes to output
            # if there is a print statement first, otherwise
            # output is empty on client.
            # TODO investigate why this is
            print(flush=True)
            output.put(f"Error occurred during input processing: {e}")
            return

        params = og.GeneratorParams(self.model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=2048)
        generator = og.Generator(self.model, params)

        response = ""

        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            response += self.tokenizer_stream.decode(new_token)

        del generator

        output.put(response)
        output.put(None)

    def generate(self, image: Optional[str], text: str) -> str:
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
            target=self.run, args=((image, text), output))
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
    '''Initial load request on start'''
    num_gpus: int = 1
    models_per_gpu: int = 1
    # Add model_id to specify which model to load
    model_id: str = "default-model-id"


@app.post("/load")
def load(request: LoadRequest):
    """
    Loads a model into memory if it's not already loaded.

    This function checks if a model specified by the request is already loaded.
    If not, it loads the model into memory and makes it available for processing.
    """

    if request.model_id not in models:
        for gpu_index in range(request.num_gpus):
            for _ in range(request.models_per_gpu):
                model = Model(
                    "..\\Phi-3-vision-128k-instruct-onnx-directml\\directml-int4-rtn-block-32",
                    gpu_index)
                model.load()
                models[request.model_id] = model
    return {"status": "200 OK", "model_id": request.model_id}


class PromptRequest(BaseModel):
    '''Definition of the client request'''
    model: int = 0
    image: Optional[str] = None
    text: str
    # Add model_id to specify which model to use
    model_id: str = "default-model-id"


@app.post("/prompt")
def prompt(request: PromptRequest):
    """
    Processes a text prompt (and optional image) using a specified model.

    This function retrieves the specified model, processes the input text and image,
    and returns the model's generated response.
    """

    if request.model_id not in models:
        raise HTTPException(
            status_code=404, detail="Model not found but tried to prompt.")

    model = models[request.model_id]

    image = None
    request_prompt = request.text
    if request.image is not None:
        image = request.image

    response_text = model.generate(image, request_prompt)

    return {"response": response_text}
