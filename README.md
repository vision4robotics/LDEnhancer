# Light Distribution-Aware Enhancement for Low-Light UAV Tracking

Code and demo videos of LDEnhancer---a low-light enhancer towards facilitating UAV tracking at night with uneven light distribution.


# Abstract 
>Visual object tracking has boosted extensive intelligent applications for unmanned aerial vehicles (UAVs). However, the state-of-the-art (SOTA) enhancer for nighttime UAV tracking always neglects the uneven light distribution in low-light images, inevitably leading to over-enhancement and saturation in bright regions. To address these issues, this work proposes a novel enhancer, i.e., LDEnhancer, enhancing nighttime UAV tracking with light distribution suppression. Specifically, a novel light distribution suppression module is designed to capture light distribution, further estimating the suppression parameter map. Meanwhile, this work proposes a new image content enhancement module to estimate the enhancement parameter map. Finally, an innovative interweave iteration adjustment is proposed for the collaborative pixel-wise adjustment of low-light images, leveraging two parameter maps. Additionally, a challenging nighttime UAV tracking benchmark with uneven light distribution, namely NUTL40, is constructed to provide a comprehensive evaluation, which contains 40 challenging sequences with over 72K frames in total. Experimental results on the public authoritative UAV benchmarks and the proposed NUTL40 demonstrate that LDEnhancer outperforms other SOTA low-light enhancers for nighttime UAV tracking. Furthermore, real-world tests on a typical UAV platform with an NVIDIA Jetson AGX Xavier confirm the practicality and efficiency of LDEnhancer. The code is available at https://github.com/vision4robotics/LDEnhancer.

<!-- ![Workflow of our tracker](https://github.com/vision4robotics/SGDViT/blob/main/imgs/2.png)

This figure shows the workflow of our tracker.
## Demo

- 📹 Demo of real-world [SGDViT](https://www.bilibili.com/video/BV1Qd4y1J7PM/?vd_source=4bf245fe6a4c3e4931ad481b87f324ae) tests.
- Refer to [Test1](https://www.bilibili.com/video/BV19e4y187Km/?vd_source=4bf245fe6a4c3e4931ad481b87f324ae) and [Test2](https://www.bilibili.com/video/BV12d4y1678S/?vd_source=4bf245fe6a4c3e4931ad481b87f324ae) on Bilibili for more real-world tests.
 -->
![Workflow of our tracker](https://github.com/vision4robotics/LDEnhancer/blob/main/workflow.png)

This figure shows the workflow of our tracker.
## Demo

- 📹 Demo of real-world tests.
- Refer to [LDEnhancer](https://youtu.be/tTI-QHqxMf8) on YouTube for more real-world tests.
## Download the datasets：
train_dataset
* [NAT2021_enhancement](https://pan.baidu.com/s/1nCHVDRDib7K9Nw1iknspeQ) (code: hjr8)

test_dataset

The dataset will be released in the coming period.

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
>2. For testing, the tracking performance is evaluated, run:
>
>     ```
>     cd SOT-test
>     python test.py
>     ```
>
>3. For training, run:
>
>     ```
>     cd LDEnhancer
>     python lowlight_train_paired.py  # FENet and IANet pretraining
>     python lowlight_train.py
>     ```



# Acknowledgements
- The code for training is constructed on [Darklighter](https://github.com/vision4robotics/DarkLighter). We sincerely thank the contribution of `Junjie Ye` for their previous work.

- The code for evaluation is implemented based on SNOT and CDT. We would like to express our sincere thanks to the contributors.

-  We sincerely thank [SiamAPN](https://github.com/vision4robotics/SiamAPN), [SiamAPN++](https://github.com/vision4robotics/SiamAPN), [SiamRPN++](https://github.com/STVIR/pysot), [SiamBAN](https://github.com/hqucv/siamban), and [SiamGAT](https://github.com/ohhhyeahhh/SiamGAT) trackers and [SCT](https://github.com/vision4robotics/SCT) and [DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension) enhancers for their efforts.


 
