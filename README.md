# Installation

```bash
# create conda environment
conda create -n geoloc python=3.10
conda activate geoloc

# install requirements
### (For most cases)
pip install -r requirements.txt         
conda install -c pytorch -c conda-forge -c nvidia faiss-gpu=1.13.1

### (For NVIDIA Jetson AGX Thor) - [ARM64 with Jetpack 7.0]
pip install -r requirements_thor.txt
conda install -c pytorch -c conda-forge -c nvidia faiss-gpu=1.9.0

### (For NVIDIA Jetson AGX Orin) - [ARM64 with Jetpack 6.2]
pip install -r requirements_orin.txt
conda install -c pytorch -c conda-forge -c nvidia faiss-gpu=1.9.0
```
```bash
# If you want to install as pip package
pip install -e .
```

## Optional 
### For running Mast3r retrieval
```bash
!!! NEED TO TEST

# install mast3r
cd ~
git clone --recursive https://github.com/naver/mast3r
echo 'export PYTHONPATH=$PYTHONPATH:~/mast3r' >> ~/.bashrc
source ~/.bashrc

# install asmk
pip install cython
git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
python setup.py build_ext --inplace
python setup.py install --skip-build
cd ..

# download mast3r checkpoints
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

# !!! make sure to change the load path in build and benchmark configs
```
### For running SegVLAD/RevisitAnything
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```