#!/bin/bash

# Set up Python environment
python3 -m venv .env
source .env/bin/activate
python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt
python3 -m pip uninstall urllib3 -y
python3 -m pip install urllib3==1.26.6
python3 -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102


# Set up data
mkdir data
cd data

# Download the Karpathy Split JSON
wget https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download= -O dataset_coco.json

python3 scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk

# bu data
mkdir data/bu_data
cd data/bu_data
echo "Downloading bu data..."
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip

echo "Extracting image features..."
python3 script/make_bu_data.py --output_dir data/cocobu

echo "All done"