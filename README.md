# Soft-NMS

This repository includes the code for Soft-NMS. Soft-NMS is integrated with two object detectors, R-FCN and Faster-RCNN. The Soft-NMS paper can be found [here](https://arxiv.org/pdf/1704.04503.pdf).

Soft-NMS is complementary to multi-scale testing and iterative bounding box regression. Check [MSRA](http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf) slides from the COCO 2017 challenge. 

**8 out of top 15 submissions used Soft-NMS in the [COCO 2017 detection challenge](http://cocodataset.org/#detections-leaderboard)!.**

We are also making our ICCV [reviews](http://www.cs.umd.edu/~bharat/reviews.html) and our [rebuttal](http://www.cs.umd.edu/~bharat/rebuttal.html) public. This should help to clarify some concerns which you may have.

To test the models with soft-NMS, clone the project and test your models as in standard object detection pipelines. This repository supports Faster-RCNN and R-FCN where an additional flag can be used for soft-NMS.

The flags are as follows,
1) Standard NMS. Use flag `TEST.SOFT_NMS` 0
2) Soft-NMS with linear weighting. Use flag `TEST.SOFT_NMS` 1 (this is the default option) 
3) Soft-NMS with Gaussian weighting. Use flag `TEST.SOFT_NMS` 2

In addition, you can specify the sigma parameter for Gaussian weighting and the threshold parameter for linear weighting. Detections below 0.001 are discarded. For integrating soft-NMS in your code, refer to `cpu_soft_nms` function in `lib/nms/cpu_nms.pyx` and `soft_nms` wrapper function in `lib/fast_rcnn/nms_wrapper.py`. You can also implement your own weighting function in this file.

For testing a model on COCO or PASCAL, use the following script

```
./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/rfcn_end2end/test_agnostic.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/rfcn_end2end_ohem_${PT_DIR}.yml \
  --set TEST.SOFT_NMS 1 # performs soft-NMS with linear weighting
  ${EXTRA_ARGS}
```

GPU_ID is the GPU you want to test on

NET_FINAL is the caffe-model to use

PT_DIR in {pascal_voc, coco} is the dataset directory

DATASET in {pascal_voc, coco} is the dataset to use

TEST_IMDB in {voc_0712_test,coco_2014_minival,coco_2014_test} is the test imdb

TEST.SOFT_NMS in {0,1,2} is flag for different NMS algorithms. 0 is standard NMS, 1 performs soft-NMS with linear weighting and 2 performs soft-NMS with gaussian weighting

Please refer to [py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU/) for details about setting up object detection pipelines.
The Soft-NMS repository also contains code for training these detectors on multiple GPUs. **The position sensitive ROI Pooling layer is updated so that interpolation of bins is correct, like ROIAlign in Mask RCNN**. The COCO detection model for R-FCN can be found [here](https://drive.google.com/open?id=0B6T5quL13CdHX04xN1ZQX2IyMms). All other detection models used in the paper are publicly available.

#### Results on MS-COCO

|                   | training data       | test data          | mAP@[0.5:0.95]   | 
|-------------------|:-------------------:|:-----------------------------:|:-----:|
|R-FCN,       NMS   | COCO 2014 train+val -minival | COCO 2015 minival     | 33.9% |
|R-FCN,  Soft-NMS L | COCO 2014 train+val -minival | COCO 2015 minival     | 34.8% |
|R-FCN,  Soft-NMS G | COCO 2014 train+val -minival | COCO 2015 minival     | 35.1% |
|F-RCNN, NMS        | COCO 2014 train+val -minival | COCO 2015 test-dev    | 24.4% |
|F-RCNN, Soft-NMS L | COCO 2014 train+val -minival | COCO 2015 test-dev    | 25.5% |
|F-RCNN, Soft-NMS G | COCO 2014 train+val -minival | COCO 2015 test-dev    | 25.5% |

R-FCN uses ResNet-101 as the backbone CNN architecture, while Faster-RCNN is based on VGG16.

### Citing Soft-NMS

If you find this repository useful in your research, please consider citing:

    @article{
      Author = {Navaneeth Bodla and Bharat Singh and Rama Chellappa and Larry S. Davis},
      Title = {Soft-NMS -- Improving Object Detection With One Line of Code},
      Booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
      Year = {2017}
    }
