# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7
ARG PYTHON_VERSION=3.12.3

FROM nvidia/cuda:12.1.1-base-ubuntu22.04 as base

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Add the NVIDIA package repositories and install the GPG key
RUN distribution=$(lsb_release -cs) && \
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
    tee /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Update package lists and install NVIDIA Container Toolkit
RUN apt-get update && \
    apt-get install -y \
    nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
RUN nvidia-ctk runtime configure --runtime=docker

# Expose the port that the application listens on.
EXPOSE 8000

EXPOSE 5678

# Run the application.
CMD uvicorn 'model_os:app' --host=0.0.0.0 --port=8000
