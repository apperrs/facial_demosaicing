# Face Image De-pixelation and Restoration Project

## Overview
In image processing, pixelation is commonly used for privacy protection, while de-pixelation aims to restore image quality. Traditional approaches primarily rely on mathematical and probabilistic models, whereas deep learning methods demonstrate superior performance through large-scale sample training. 

This project innovatively proposes a **model cascading approach** that combines CycleGAN with DCEDN (Deep Convolutional Encoder-Decoder Network), effectively restoring facial pixelated images while addressing color shift issues.

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
opencv-python pillow torch torchvision torchaudio \
torchsummary onnxruntime torch-optimizer ntpath matplotlib dominate

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

## 📬 Contact

For inquiries, collaborations, or feedback:  
✉️ [Email the maintainer](mailto:apperrs@gmail.com)  
💻 Open to contributions and research collaborations  


