# Enhanced Facial De-pixelization: A Cascaded CycleGAN and DCEDN Approach

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16441619.svg)](https://doi.org/10.5281/zenodo.16441619)

## Overview
In image processing, pixelation is commonly used for privacy protection, while de-pixelation aims to restore image quality. Traditional approaches primarily rely on mathematical and probabilistic models, whereas deep learning methods demonstrate superior performance through large-scale sample training. 

This project innovatively proposes a **model cascading approach** that combines CycleGAN with DCEDN (Deep Convolutional Encoder-Decoder Network), effectively restoring facial pixelated images while addressing color shift issues.

📢 **Citation Request:** If you use this code or reference our work, please consider citing our paper currently under review at *The Visual Computer*.

**Key Features:**
- Dual-stage architecture combining CycleGAN and DCEDN
- Effective solution for color shift problems
- High-quality facial image restoration
- Step-wise training strategy

## Dataset
We use the **CelebA dataset** (Large-scale CelebFaces Attributes Dataset) for facial image de-pixelation and restoration research.

**Dataset Specifications:**
- 202,599 celebrity images
- 40 attribute annotations per image
- Diverse pose variations and complex backgrounds
- Official Website: [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Environment Requirements
To run this project, you'll need the following Python packages: 
```bash
opencv-python pillow torch torchvision torchaudio \
torchsummary onnxruntime torch-optimizer ntpath matplotlib dominate
```
Set up your environment using:
```bash
pip install -r requirements.txt
```

## Training Process

### 1. Train CycleGAN Model
```bash
python3 train_cyclegan.py
```
### 2. Train DCEDN Model
First convert the CycleGAN model format:
```bash
python3 convert_cyclegan_pt2onnx.py
```
Then train the DCEDN model:
```bash
python3 train_dcedn.py
```

## Prediction
Run the prediction script:
```bash
python3 predict.py
```

## Key Contributions

- **Novel Cascaded Architecture**: First to propose combining CycleGAN with DCEDN for facial image de-pixelation
- **Color Correction System**: DCEDN module specifically designed to address color shift artifacts in CycleGAN outputs
- **Optimized Training Pipeline**: Developed an effective step-by-step training strategy for both models
- **Production-Ready Implementation**: Complete workflow from data processing to final prediction

## Citation
Please cite this repository and related paper as:
```bibtex
@misc{facial_depixel_2024,
  author       = {Heng Zhang, YiJie Xue, Yanli Liu},
  title        = {Enhanced Facial De-pixelization: A Cascaded CycleGAN and DCEDN Approach},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16441619},
  url          = {https://doi.org/10.5281/zenodo.16441619}
}
```

## 📬 Contact

For inquiries, collaborations, or feedback:  
✉️ [Email the maintainer](mailto:apperrs@gmail.com)  
💻 Open to contributions and research collaborations  


