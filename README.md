# Installation

```bash
# create conda environment
conda create -n geoloc python=3.12
conda activate geoloc

# install requirements
### (For most cases)
pip install -r requirements.txt         
conda install -c pytorch -c conda-forge -c nvidia faiss-gpu=1.9.0

### (For NVIDIA Jetson AGX Thor) - [ARM64 with Jetpack 7.0 and CUDA 13.0]
pip install -r requirements_thor.txt
conda install -c conda-forge -c nvidia faiss-gpu=1.9.0

### (For NVIDIA Jetson AGX Orin) - [ARM64 with Jetpack 6.2 and CUDA 12.6]
pip install -r requirements_orin.txt
conda install -c conda-forge -c nvidia faiss-gpu=1.9.0
```
```bash
cd geoloc/third_party/LoMa/
pip install -e .

cd geoloc/third_party/RoMa/
pip install -e .

cd geoloc
pip install .
```

```python
### Clone EUPE repository
git clone org-16943930@github.com:facebookresearch/EUPE.git
```
Make sure to also change the `repo_dir` and `weights_path` in the `/models/vpr/eupe_salad_20260428_141149/train_config.yaml` to the appropriate values. Additionally download the [`EUPE-ViT-B.pt`](https://huggingface.co/facebook/EUPE-ViT-B/) weights

### **! ! ! NOTE - Installing torch and torchvision on AGX Orin ! ! !**
https://forums.developer.nvidia.com/t/pytorch-2-8-0-on-jetson-orin-nano-importerror-libcudss-so-0-not-found/346195

The prebuilt wheels for newer version of torch and torchvision (>2.8.0) for Jetpack 6.2, CUDA 12.6, don't include the CuDSS library, so you have to install it manully following [this steps](https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_local).

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