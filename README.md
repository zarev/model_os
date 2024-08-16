# Model_OS

`model_os` is your sentient OS.

## Features

- **Load ONNX Models**: Load models into memory on application startup or on demand.
- **Process Inputs**: Handle and process text and image inputs using loaded models.
- **Asynchronous Support**: Manage application lifecycle with asynchronous functions.
- **Multi-GPU Support**: Load models across multiple GPUs.
- **Customizable Model Prompts**: Generate custom prompts for model input.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zarev/model_os.git
   cd model_os
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Install the required packages listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the FastAPI application with Uvicorn, execute:

```bash
uvicorn model_os:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`.

## API Endpoints

### `/load` (POST)
Load a model into memory. This endpoint can be used to initialize models on application startup.

**Request Body**:
```json
{
    "num_gpus": 1,
    "models_per_gpu": 1,
    "model_id": "default-model-id"
}
```

### `/prompt` (POST)
Process a text prompt (and optionally an image) using a specified model.

**Request Body**:
```json
{
    "model": 0,
    "image": "optional-image-path",
    "text": "Your text prompt here",
    "model_id": "default-model-id"
}
```

## Code Overview

### Main Components

- **Model Class**: Represents an ONNX model, with methods to load the model and process inputs.
- **FastAPI Application**: Defines routes for loading models and processing prompts.
- **Asynchronous Lifespan Manager**: Manages the startup and shutdown processes for the FastAPI application.

### Important Files

- `model_os.py`: The main module containing the FastAPI application, model loading logic, and input processing functions.

## Customization

You can customize the following aspects of the application:

- **Model Paths**: Update model paths and IDs in the `LoadRequest` and `Model` classes.
- **Prompt Templates**: Modify the `template_prompt` in the `run` method of the `Model` class for custom input processing.

## Development and Contribution

Feel free to fork the repository and submit pull requests. Contributions are welcome!

### Setting Up Development Environment

1. **Install Development Dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run Linting and Tests**:
   ```bash
   pylint model_os.py
   pytest
   ```

## License

This project is licensed under the MIT License.

---