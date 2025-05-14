# MakeControlVideo

This repository is making video dataset for Training WAN2.1 Fun Control.  
Write youtube URL in ```jsonfiles/proto.json``` and run make_youtube_video_data.py.

## Requirements

Install ffmpeg to run 
```
sudo apt-get update
sudo apt install ffmpeg
```
## Installation
(recommend Env : python 3.10 , CUDA 12.6)

make Virtual Env and activate

Install pytorch [here](https://pytorch.org/)

run ```pip install -r requirements.txt```


## Settings
add ```.env``` file in your dir
and write
```
BATH_PATH = <path to This repository>
LLM_PATH = <path to LLM dir like ~/ComfyUI/models/LLM>
```
Download Qwen 2.5 VL models from [here](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

and put all of them in ```Qwen2_5-VL-7B-Instruct``` and put that dir unger ```LLM_PATH```