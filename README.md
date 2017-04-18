# Soft-NMS

## Improving Object Detection With One Line of Code

This repository includes the code for Soft-NMS. It is also integrated with two object detectors, R-FCN and Faster-RCNN. Soft-NMS paper can be found [here](https://arxiv.org/pdf/1704.04503.pdf).

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

GPU_ID is the GPU you want to train on
NET_FINAL is the caffe-model to use
PT_DIR in {pascal_voc, coco} 
DATASET in {pascal_voc, coco} is the dataset to use
TEST_IMDB in {voc_0712_test,coco_2014_val} is the test imdb
TEST.SOFT_NMS in {0,1,2}. 0 is standard NMS, 1 perform soft-NMS with linear weighting and 2 performs soft-NMS with gaussian weighting

Please refer to [py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU/) for details about setting up object detection pipelines.
This repository also contains code for training these detectors on multiple GPUs. The COCO detection model for R-FCN can be found [here](https://drive.google.com/file/d/0B6T5quL13CdHMGtMUWFFSXd2Ym8). All other detection models are publicly available.

#### Results on MS-COCO

|                   | training data       | test data          | mAP@[0.5:0.95]   | 
|-------------------|:-------------------:|:-----------------------------:|:-----:|
|R-FCN,       NMS   | COCO 2014 train+val -minival | COCO 2015 test-dev    | 31.1% |
|R-FCN,  Soft-NMS L | COCO 2014 train+val -minival | COCO 2015 test-dev    | 32.2% |
|R-FCN,  Soft-NMS G | COCO 2014 train+val -minival | COCO 2015 test-dev    | 32.4% |
|F-RCNN, NMS        | COCO 2014 train+val -minival | COCO 2015 test-dev    | 24.4% |
|F-RCNN, Soft-NMS L | COCO 2014 train+val -minival | COCO 2015 test-dev    | 25.5% |
|F-RCNN, Soft-NMS G | COCO 2014 train+val -minival | COCO 2015 test-dev    | 25.5% |

R-FCN uses ResNet-101 as the backbone CNN architecture, while Faster-RCNN is based on VGG16.

### Citing Soft-NMS

If you find this repository useful in your research, please consider citing:

    @article{1704.04503,
      Author = {Navaneeth Bodla and Bharat Singh and Rama Chellappa and Larry S. Davis},
      Title = {Improving Object Detection With One Line of Code},
      Journal = {arXiv preprint arXiv:1704.04503},
      Year = {2017}
    }
