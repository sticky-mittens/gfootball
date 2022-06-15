# Configure Docker -- In order to see rendered game you need to allow Docker containers access X server:
xhost +"local:docker@"

# build docker image 
## Tensorflow without GPU-training support version (enough for M1 mac)
docker build --build-arg DOCKER_BASE=ubuntu:20.04 . -t gfootball

# ## Tensorflow with GPU-training support version
# docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 . -t gfootball

# Start the Docker image 
## If you get errors related to --gpus all flag, you can replace it with --device /dev/dri/[X] adding this flag 
## for every file in the /dev/dri/ directory. It makes sure that GPU is visibile inside the Docker image. You can 
## also drop it altogether (environment will try to perform software rendering). 
docker run --gpus all -e DISPLAY=$DISPLAY -it -v /tmp/.X11-unix:/tmp/.X11-unix:rw gfootball bash

## Run env -- Inside the Docker image you can interact with the environment the same way as in case of local installation.

# pip3 install numpy matplotlib
# pip3 install torch torchvision torchaudio