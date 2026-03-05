<h2 align="center" width="100%">
[ArXiv'26] ArtHOI: Articulated Human-Object Interaction Synthesis by 4D Reconstruction from Video Priors
</h2>
<div>
<div align="center">
    <a href='https://inso-13.github.io/' target='_blank'>Zihao Huang</a><sup>1,2,3</sup>&emsp;
    <a href='http://tqtqliu.github.io/' target='_blank'>Tianqi Liu</a><sup>1,2,3</sup>&emsp;
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a><sup>2</sup>&emsp;
    <a href='https://daniellli.github.io/' target='_blank'>Shaocong Xu</a><sup>3</sup>&emsp;
    <a href='https://sainingzhang.github.io/' target='_blank'>Saining Zhang</a><sup>2,3</sup><br>
    <a href='https://li-xingxiao.github.io/homepage/' target='_blank'>Lixing Xiao</a><sup>5</sup>&emsp;
    <a href='http://faculty.hust.edu.cn/caozhiguo1/en/index.htm/' target='_blank'>Zhiguo Cao</a><sup>1</sup>&emsp;
    <a href='https://weivision.github.io/' target='_blank'>Wei Li</a><sup>2</sup>&emsp;
    <a href='https://sites.google.com/view/fromandto' target='_blank'>Hao Zhao</a><sup>4,3</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>2</sup>
</div>
<div>
<div align="center">
    <sup>1</sup>Huazhong University of Science and Technology&emsp;
    <sup>2</sup>Nanyang Technological University<br>
    <sup>3</sup>Beijing Academy of Artificial Intelligence&emsp;
    <sup>4</sup>AIR, Tsinghua University&emsp;
    <sup>5</sup>Zhejiang University<br>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2603.04338" target='_blank'>
    <img src="http://img.shields.io/badge/cs.CV-arXiv%3A2603.04338-B31B1B.svg" alt="ArXiv">
  </a>
  <a href="https://arthoi.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project Page-%F0%9F%93%9a-lightblue" alt="Project Page">
  </a>
  <a href="https://youtu.be/0gzpn4AKJBM">
    <img src="https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white">
  </a>
  <a href="#">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=Inso-13.ArtHOI" alt="Visitors">
  </a>
</p>

>**TL;DR**: <em>ArtHOI enables zero-shot synthesis of realistic human interactions with articulated objects.</em>

## 🔨 Environments
```bash
# Assuming CUDA 11.7
conda create -n arthoi python=3.9
conda activate arthoi
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# FRNN
git clone --recursive https://github.com/lxxue/FRNN.git
cd FRNN/external/prefix_sum
pip install .
cd ../../
pip install -e .

pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/facebookresearch/co-tracker.git
pip install git+https://github.com/yzslab/simple-knn.git
pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

## 🚀 Startup

1. Download the required model files and organize them in the `assets` folder:
```bash
assets/
├── sam_vit_h_4b8939.pth              # SAM2 model from [SAM2](https://github.com/facebookresearch/sam2)
└── body_models/smplx/                # SMPL-X models from [SMPL-X](https://smpl-x.is.tue.mpg.de/)
     ├── SMPLX_NEUTRAL.npz             
     ├── SMPLX_MALE.npz                
     ├── SMPLX_FEMALE.npz              
     └── smplx_vert_segmentation.json  
```

2. Download the [demo data](https://drive.google.com/file/d/17CqcVwOM4khGWyERAwxAO48ruMG869YA/view?usp=sharing) and put it in the `data` folder:
```bash
data/
└── {scene_name}/
```

3. Train the model:
```bash
cd src
conda activate arthoi
python train.py --scene {scene_name}
# For example
python train.py --scene open-cabinet
```

## 📊 Results
The results will be saved in the `results` folder.
```bash
results/{scene_name}/arthoi/
├── params/          # Model parameters and optimization states
├── renders/         # Generated video frames and visualizations
```

### 📦 Custom Data
If you want to train the model on your own data, you can prepare the data in the following format:
```
data/
├── {your_scene_name}/
│   ├── init_params/
│   │   ├── align.json               # Human, Object, and Camera alignment
│   │   ├── smplx.json               # SMPL-X parameters from [GVHMR](https://github.com/zju3dv/GVHMR)
│   │   ├── hamer.json               # Hand parameters from [HAMER](https://github.com/geopavlakos/hamer)
│   │   └── camera.json              # Camera intrinsics and extrinsics
│   ├── init_gaussians/
│   │   ├── human_cano.ply           # Human canonical mesh
│   │   ├── human.ply                # Human mesh at first frame
│   │   ├── object.ply               # Object mesh at first frame
│   │   └── scene.ply                # Scene mesh
│   └── priors/
│       ├── images/                  # Extracted frames
│       ├── human_masks/             # Human masks from [SAM2](https://github.com/facebookresearch/sam2)
│       ├── object_masks/            # Object masks from [SAM2](https://github.com/facebookresearch/sam2)
│       ├── cotracker/               # Dense correspondences from [CoTracker](https://github.com/facebookresearch/co-tracker)
│       └── hmr4d_results.pt         # 4D human motion from [GVHMR](https://github.com/zju3dv/GVHMR)
```


## 📝 Citation
If you find our work useful for your research, please cite our paper.

```
  @article{huang2026arthoi,
    title={ArtHOI: Articulated Human-Object Interaction Synthesis by 4D Reconstruction from Video Priors},
    author={Huang, Zihao and Liu, Tianqi and Chen, Zhaoxi and Xu, Shaocong and Zhang, Saining and Xiao, Lixing and Cao, Zhiguo and Li, Wei and Zhao, Hao and Liu, Ziwei},
    journal={arXiv preprint arXiv:2603.04338},
    year={2026}
  }
```