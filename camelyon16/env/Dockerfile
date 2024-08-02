# Use the base image with CUDA support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set environment variables to configure tzdata non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Install system packages and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

# Check Python and pip installation
RUN python3 --version && pip3 --version

# Install additional Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python-headless \
    scikit-image \
    openslide-python \
    matplotlib \
    pillow

# Install additional Python packages
RUN pip3 install --no-cache-dir \
    torch==1.13.0 torchvision==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu117 \
    scikit-image \
    openslide-python \
    matplotlib \
    pillow

# Set the command to run when the container starts
CMD ["bash"]