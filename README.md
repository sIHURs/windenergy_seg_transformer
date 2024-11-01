# windenergy_seg_transformer
Semi-Supervised Learning (Cross Teaching CNN & Transformer) in the image segmentation task for wind energy.

# Result - mIoU
--testset--
Cross Teaching Unet: 0.9383
Supervised training Unet: 0.9369
Cross Teaching Swin-Unet: 0.9160
Supervised training Swin-Unet: 0.9136

--trainset--
Cross Teaching Unet: 0.9452
Supervised training Unet: 0.9444
Cross Teaching Swin-Unet: 0.9468
Supervised training Swin-Unet: 0.9413

This work is modified from [Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer](https://arxiv.org/abs/2112.04894), [github](https://github.com/HiLab-git/SSL4MIS)
