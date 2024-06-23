# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile
FROM python:3.8-slim

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /code

RUN apt-get update -y
RUN apt install git --fix-missing -y
RUN apt install wget -y

# installing other libraries
RUN apt-get install python3-pip -y && \
    apt-get -y install sudo
RUN apt-get install curl -y
RUN apt-get update && apt-get install -y git
RUN apt-get install libblas-dev -y && apt-get install liblapack-dev -y
RUN apt-get install gfortran -y
RUN apt-get install libpng-dev -y
RUN apt-get install python3-dev -y
RUN apt install wget -y && \
    apt install unzip

# necessary for opencv in docker
RUN apt-get install -y ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6

WORKDIR /code

# install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Download pretrained models
RUN mkdir -p models
RUN wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/densenet121-in1k.pth.tar" -O models/densenet121-in1k.pth.tar && \
    wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/swin-in22k.pth.tar" -O models/swin-in22k.pth.tar && \
    wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/swin-simmim.pth.tar" -O models/swin-simmim.pth.tar && \
    wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/vit-in1k.pth.tar" -O models/vit-in1k.pth.tar


# Download example images
RUN mkdir example_images
RUN wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/00010575_002.png" -O example_images/00010575_002.png && \
    wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/00010828_039.png" -O example_images/00010828_039.png && \
    wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/00011925_072.png" -O example_images/00011925_072.png && \
    wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/00018253_059.png" -O example_images/00018253_059.png && \
    wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/00020482_032.png" -O example_images/00020482_032.png && \
    wget "https://github.com/taheeraahmed/master-thesis/releases/download/trained-models/00026221_001.png" -O example_images/00026221_001.png

# CMD ["/bin/bash"]
CMD ["python3", "src/app.py"]