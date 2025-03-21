# One-for-All: Towards Universal Domain Translation with a Single StyleGAN

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://zhanjiahui.github.io/UniTranslator/)  
**Paper Title**: One-for-All: Towards Universal Domain Translation with a Single StyleGAN  
**Authors**: Yong Du*, Jiahui Zhan*, Xinzhe Li, Junyu Dong, Sheng Chen, Ming-Hsuan Yang, Shengfeng He  
**Published in**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025  
**DOI**: [10.1109/TPAMI.2025.3530099](https://ieeexplore.ieee.org/document/10848371)  
*(\* denotes equal contribution)*

---

## Getting Started

#### 1. Environment Setup
1.1 Create Conda Environment and Install PyTorch

```bash
conda create -n Unitranslator python=3.7
conda activate Unitranslator
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
1.2 Install CLIP

👉 Follow the official instructions from the [CLIP repository](https://github.com/openai/CLIP)  

#### 2. Download Pre-trained Models
2.1 Download `stylegan2-ffhq-config-f.pkl` from [StyleGAN2 Official Models](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/).

2.2 Convert `.pkl` to `.pt` using [rosinality's StyleGAN2-PyTorch](https://github.com/rosinality/stylegan2-pytorch).

2.3 Place the converted `stylegan2-ffhq-config-f.pt` in the root directory of this repo.

#### 3. Run Translation
```bash
cd One-for-All
CUDA_VISIBLE_DEVICES=0 python run.py \
  -duplicates 1 \
  -input_dir examples \
  -output_dir output \
  -eps 0.02 \
  -tile_latent \
  -loss_str "10*L2+1*LPIPS+1*P" \
  -loss_str2 "10*L2+1*LPIPS+1*P" \
  -steps 71 \
  -fc_every 1 \
  -seed 2
```

---

## Key References
**[1]** PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models. In CVPR2020.

**[2]** Improved StyleGAN Embedding: Where are the Good Latents? arXiv:2012.09036 (2020).


---

## Implementation Notes
This implementation is heavily adapted from the [PULSE repository](https://github.com/alex-damian/pulse). We gratefully acknowledge the work of the original authors.

---

## Citation
```bibtex
@ARTICLE{10848371,
  author={Du, Yong and Zhan, Jiahui and Li, Xinzhe and Dong, Junyu and Chen, Sheng and Yang, Ming-Hsuan and He, Shengfeng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={One-for-All: Towards Universal Domain Translation With a Single StyleGAN}, 
  year={2025},
  volume={47},
  number={4},
  pages={2865-2881},
  doi={10.1109/TPAMI.2025.3530099}}
```

