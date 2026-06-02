# HMMF
A Hybrid Space Model for Misaligned Multi-modality Image Fusion[AAAI2026]

# Data preparation
- You can generate misaligned infrared-visible images for training/testing by
```
   cd ./data
   python generate_affine_deform_data.py
```
- You can obtain saliency masks for training the fusion process of infrared and visible images by
```
   cd ./data
   python get_sam_mask.py
```

# Run
```
   python train.py
   python test.py
```

# Citation
```
@article{HMMF2026AAAI, 
   title={A Hybrid Space Model for Misaligned Multi-modality Image Fusion}, 
   author={Xiao, Yi and Wang, Jia and Liu, Zhu and Wang, Di and Liu, Jinyuan and Liu, Risheng}, 
   volume={40},
   number={13}, 
   journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
   year={2026}, 
   pages={11005–11013} }
```
