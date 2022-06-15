# Setup

* CONDA on Linux
```
# install system packages in linux distro
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

# create conda env
conda create -n grf_env python=3.7 anaconda
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# clone original repo and install gfootball and reqs
git clone https://github.com/google-research/football.git
cd football
python3 -m pip install --upgrade pip setuptools psutil wheel
python3 -m pip install .

# clone our repo
cd ../
git clone https://github.com/sticky-mittens/gfootball.git
cd gfootball
```

* DOCKER

```
# clone original repo and install docker container
git clone https://github.com/google-research/football.git
xhost +"local:docker@"
# For GPU support in docker
docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 . -t gfootball 
# Without GPU do the following
# docker build --build-arg DOCKER_BASE=ubuntu:20.04 . -t gfootball

# Start docker image and install some reqs
docker run --gpus all -e DISPLAY=$DISPLAY -it -v /tmp/.X11-unix:/tmp/.X11-unix:rw gfootball bash
pip3 install numpy matplotlib
pip3 install torch torchvision torchaudio

# clone our repo inside docker container
cd ../
git clone https://github.com/sticky-mittens/gfootball.git
cd gfootball
```

# Our Code

```
bash best_run_1.sh
```
Wait for above to finish and then run,
```
bash best_run_2.sh
```

# Generate videos of agents while training
```
bash video_run_2.sh
```

# Baseline

```
bash run_baseline.sh
```