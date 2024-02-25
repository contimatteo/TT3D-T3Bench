#!/bin/bash

pip install -U pip wheel
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
pip install ninja

pip install pyrender
pip install openai==0.28.1
pip install open3d trimesh Pillow backoff tqdm matplotlib
pip install image-reward
pip install git+https://github.com/openai/CLIP.git

# pip install -q omegaconf iopath decord webdataset einops pycocoevalcap
# pip install --no-deps salesforce-lavis
pip install salesforce-lavis

pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 
pip install opencv-python==4.7.0.72
pip install python-dotenv spacy
pip install transformers pillow
