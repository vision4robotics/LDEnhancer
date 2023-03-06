# Light Distribution-Aware Enhancement for Low-Light UAV Tracking

Code and demo videos of LDEnhancer---a low-light enhancer towards facilitating UAV tracking in the dark.


# Abstract 
>Visual object tracking has demonstrated extensive applications for autonomous unmanned aerial vehicles (UAVs).  However, low-Light images captured by UAVs suffer from object information loss due to uneven light distributions. The state-of-the-art (SOTA) enhancement methods always neglect the nighttime light distribution, which inevitably leads to over-enhancement and over-saturation for the bright regions of the low-light images and UAV tracking failure. To address these issues, this work proposes a novel light distribution-aware low-light image enhancer, i.e. , LDEnhancer. Specifically, low-light image enhancement is treated as a task of collaborative pixel-wise adjustment with parameter map estimation. The proposed method, namely LDEnhancer, employs an illumination-prior suppression module to estimate the light distribution and predict the illumination suppression parameter map. Meanwhile, a content-centric guidance module is designed to increase the attention to content information and calculate the content enhancement parameter map in accordance with the estimated light distribution. Finally, a novel interweaves iteration enhancement method is proposed for the iterative enhancement of low-light images. Experimental results on the public authoritative UAV benchmarks demonstrate that LDEnhancer outperforms other SOTA low-light enhancers. Furthermore, real-world tests on a typical UAV platform confirm the practicality and efficiency of LDEnhancer in low-light UAV tracking. The source code and demo videos can be accessed at https://github.com/vision4robotics/LDEnhancer.

# Demo video



# Contact 
Liangliang Yao

Email: 1951018@tongji.edu.cn

Changhong Fu

Email: changhongfu@tongji.edu.cn

# Demonstration running instructions

### Requirements

1.Python 3.7

2.Pytorch 1.0.0

3.opencv-python

4.torchvision

5.cuda 10.2

>Download the package, extract it and follow two steps:
>
>1. Put test images in data/test/, put training data in data/train/.
>
>2. For testing, run:
>
>     ```
>     python lowlight_test.py
>     ```
>
>3. For training, run:
>
>     ```
>     python lowlight_train_1.py  # FENet and IANet pretraining
>     python lowlight_train.py
>     ```



# Acknowledgements

We sincerely thank the contribution of `Chongyi Li`, `Junjie Ye` for their previous work [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) and [Darklighter](https://github.com/vision4robotics/DarkLighter).

