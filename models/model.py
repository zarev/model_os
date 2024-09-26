# models/model.py

import os
import threading
import queue
import base64
from typing import Optional, Tuple
from io import BytesIO

from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

# Import process_audio from utils
from utils import process_audio


class Model:
    """
    A class representing an inference model.

    Attributes:
    -----------
    model : transformers.PreTrainedModel
        The loaded model instance.
    processor : transformers.PreTrainedProcessor
        The processor associated with the model.
    model_path : str
        The file path to the model.
    model_type : Type
        The class type of the model to load.
    use_safetensors : Optional[bool]
        Whether to use safetensors.
    model_name : str
        The name of the model.
    revision : Optional[str]
        The revision identifier of the model.
    loaded : bool
        A flag indicating whether the model is currently loaded.

    Methods:
    --------
    load():
        Loads the model and initializes it.

    generate(image: Optional[str], audio: Optional[str], text: str) -> str:
        Processes the input data and generates a response.
    """

    def __init__(self, model_path: str, index: int):
        """
        Initializes the Model class with a given model path and index.

        Parameters:
        -----------
        model_path : str
            The file path to the model.
        index : int
            The index identifying the model instance.
        """
        self.model = None
        self.processor = None
        self.model_path = model_path
        self.index = index
        self.loaded = False
        self.model_type = None
        self.use_safetensors = None
        self.model_name = None
        self.revision = None
        self.setup_model_parameters()

    def setup_model_parameters(self):
        """
        Sets up model-specific parameters based on the model path.
        """
        if 'distil' in self.model_path:
            # WHISPER model settings
            self.model_type = AutoModelForSpeechSeq2Seq
            self.use_safetensors = True
            self.model_name = "distil-whisper/distil-large-v3"
            self.revision = "871351a"
        else:
            # PHI model settings
            self.model_type = AutoModelForCausalLM
            self.use_safetensors = None
            self.model_name = "microsoft/Phi-3-vision-128k-instruct"
            self.revision = "c45209e"

    def load(self):
        """
        Loads the model and processor.
        """
        #config_exists = os.path.exists(os.path.join(self.model_path, 'config.json'))

        #if not config_exists:
        #    print(f"Config file not found in {self.model_path}")
        #    return

        self.model = self.model_type.from_pretrained(
            self.model_path,
            revision=self.revision,
            cache_dir=self.model_path,
            use_safetensors=self.use_safetensors,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
            local_files_only=True
        )

#working        print('MODEL INIT MODEL', self.model)

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            revision=self.revision,
            cache_dir=os.path.dirname(self.model_path),
            local_files_only=True
        )
#working        print('MODEL INIT PROCESSOR', self.processor)

        self.loaded = True

    def generate(self, image: Optional[str], audio: Optional[str], text: str) -> str:
        """
        Processes the input data and generates a response.

        Parameters:
        -----------
        image : Optional[str]
            Base64-encoded image string, or None.
        audio : Optional[str]
            Base64-encoded audio string, or None.
        text : str
            The text prompt to be processed.

        Returns:
        --------
        str
            The generated response from the model.
        """
        output_queue = queue.Queue()
        thread = threading.Thread(
            target=self._run, args=((image, audio, text), output_queue))
        thread.daemon = True
        thread.start()

        result = []
        while thread.is_alive():
            try:
                output_text = output_queue.get(timeout=5)
                if output_text is None:
                    break
                result.append(output_text)
            except queue.Empty:
                continue

        thread.join()
        return "".join(result)

    def _run(self, input_data: Tuple[Optional[str], Optional[str], str], output: queue.Queue) -> None:
        """
        Internal method to process input data and generate model response.

        Parameters:
        -----------
        input_data : Tuple[Optional[str], Optional[str], str]
            A tuple containing image data, audio data, and text.
        output : queue.Queue
            A queue to store the generated response.
        """
        input_image, input_audio, input_text = input_data

        if input_audio:
            audio_samples = process_audio(input_audio)
            if self.model_type == AutoModelForSpeechSeq2Seq:
                # WHISPER model processing
                input_features = self.processor(
                    audio_samples,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True
                )
                input_features = input_features.to("cuda", dtype=torch.float16)

                # Model inference
                pred_ids = self.model.generate(input_features.input_features, max_new_tokens=440)
                # Decode the predicted IDs to get the transcription
                pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

                # Return the transcription directly
                output.put(pred_text)
                output.put(None)
                return
            else:
                # For PHI model, append the transcription to the input text
                input_text += " " + self._transcribe_audio(audio_samples)

        messages = [{
            "role": "user",
            "content": ''
        }]

        image_obj = None
        # If there is an image, load it and add tag to prompt
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
            inputs = self.processor(
                templated_prompt,
                images=image_obj,
                return_tensors="pt"
            ).to("cuda:0")

            generation_args = {
                "max_new_tokens": 2048,
                "temperature": 0.1,
                "do_sample": True,
            }

            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **generation_args
            )

            # Remove input tokens
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

            output.put(response)
            output.put(None)

        except Exception as e:
            output.put(f"Error occurred during model processing: {e}")
            output.put(None)

    def _transcribe_audio(self, audio_samples):
        """
        Transcribes audio samples using the WHISPER model.

        Parameters:
        -----------
        audio_samples : numpy.ndarray
            The audio samples to transcribe.

        Returns:
        --------
        str
            The transcribed text.
        """
        # Implement the transcription using a loaded WHISPER model
        # This requires that you have a WHISPER model instance available
        # For the purpose of this example, we'll assume you have a method to do this
        # You might need to adjust this method based on your actual implementation
        pass  # Replace with actual transcription code or method call
