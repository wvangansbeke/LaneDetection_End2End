# LaneDetection_End2End
This repo contains the implementation of our paper [End-to-end Lane Detection through Differentiable Least-Squares Fitting](https://arxiv.org/abs/1902.00293v1). 
## Introduction
This repository compares two methods to achieve higher accuracy for lane detection applications. The former is the conventional segmentation approach and the latter will tackle this problem in an end-to-end manner. The segmentation approach depends on the cross-entropy loss in order to learn the road markings by attention. However this approach is not necessarily the most accurate. Since the final line coordinates are desired, we show that a complete end-to-end method achieves better results. Code will be released soon.

![end_to_end](https://user-images.githubusercontent.com/9694230/51836593-12459400-2301-11e9-9d1b-37cbe936f8cc.gif)
