## Installation
Take the same installation steps as maskrcnn-benchmark, since our repository is modified from this one.

But we recommend to make these two modification to avoid some unexpected errors:

* Change pillow version to 5.4.1
    ```
    pip install pillow==5.4.1
    ```

* Install torchvision by
    ```bash
    git clone pytorch/vision
    cd vision
    git checkout 98ca
    python setup.py install
    ```



### Requirements:
- PyTorch 1.0 from a nightly release. It **will not** work with 1.0 nor 1.0.1. Installation instructions can be found in https://pytorch.org/get-started/locally/
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

```

### Option 2: Docker Image (Requires CUDA, Linux only)

We provide a docker image to test our model:

```bash
docker pull djiajun1206/pytorch_1.1_nightly_torchvision_98ca
```

