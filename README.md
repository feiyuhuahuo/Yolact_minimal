## Yolact_minimal
Minimal PyTorch implementation of [Yolact:《YOLACT: Real-time Instance Segmentation》](https://arxiv.org/abs/1904.02689).  
The original project is [here](https://github.com/dbolya/yolact).  

This implementation simplified the original code, preserved the main function and made the network easy to understand.   

Following instruction is based on resnet-101.  
### The network structure.  
![Example 0](data/network.png)

## Environments
PyTorch >= 1.1.
Python >= 3.6.
Other common packages.   

## Prepare
- Download COCO 2017 datasets, modify the paths of training and evalution datasets in `data/config.py`.
- If some directories are missed, just create them by yourself.   
- Download weights.

Yolact trained weights.
| Backbone  | box mAP  | mask mAP  | Google Drive                                                                                                         |Baidu Cloud                                                        |
|:---------:|:--------:|:---------:|:--------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------:|
| Resnet50  | 30.25    | 28.04     | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)  | [password: mksf](https://pan.baidu.com/s/1XDeDwg1Xw9GJCucJNqdNZw) |
| Resnet101 | 32.54    | 29.83     | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)      | [password: oubr](https://pan.baidu.com/s/1uX_v1RPISxgwQ2LdsbJrJQ) |

ImageNet pre-trained weights.
| Backbone  | Google Drive                                                                                                    |Baidu Cloud                                                        |
|:---------:|:---------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------:|
| Resnet50  | [resnet50-19c8e357.pth](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing)     | [password: a6ee](https://pan.baidu.com/s/1aFLE-e1KdH_FxRlisWzTHw) |
| Resnet101 | [resnet101_reducedfc.pth](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing)   | [password: kdht](https://pan.baidu.com/s/1ha4aH7xVg-0J0Ukcqcr6OQ) |


## Train
```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config
# Training with different batch_size (remember to set freeze_bn=True in `config.py` when the batch_size is smaller than 4).
python train.py --config=yolact_base_config --batch_size=4
# Training with different image size (anchor settings related to image size will be adjusted automatically).
python train.py --config=yolact_base_config --img_size=400
# Resume training with the latest trained model.
python train.py --config=yolact_base_config --resume latest
# Resume training with a specified model.
python train.py --config=yolact_base_config --resume yolact_base_2_35000.pth
# Set evalution interval during training.
python train.py --config=yolact_base_config --val_interval 20000
```

## Evalution
```Shell
# Evaluate on COCO val2017 (configs will be parsed according to the model name).
python eval.py --trained_model=yolact_base_54_800000.pth
# Evaluate with a specified number of images.
python eval.py --trained_model=yolact_base_54_800000.pth --max_num=1000
```
The results should be:
![Example 1](data/mAP.png)

```Shell
# Create a json file and then use the COCO API to evaluate the COCO detection result.
python eval.py --trained_model=yolact_base_54_800000.pth --cocoapi
# Benchmark
python eval.py --trained_model=yolact_base_54_800000.pth --benchmark --max_num=1000
```
## Detect
![Example 2](data/2.jpg)
```Shell
# Detect images, pass the path of your image directory to --image.
python detect.py --trained_model=yolact_base_54_800000.pth --image images
# Detect a video, pass the path of your video to --video.
python detect.py --trained_model=yolact_base_54_800000.pth --video video/1.mp4
# Use --real_time to detect real-timely.
python detect.py --trained_model=yolact_base_54_800000.pth --video video/1.mp4 --real_time
# Use --hide_mask, --hide_score, --show_lincomb and so on to get different results.
python detect.py --trained_model=yolact_base_54_800000.pth --image images --hide_mask
```

## Train on PASCAL_SBD datasets
- Download PASCAL_SBD datasets from [here](http://home.bharathh.info/pubs/codes/SBD/download.html), modify the path of `img` folder in `data/config.py`.
- Then,
```Shell
python utils/pascal2coco.py
```
- Download the pre-trained weights.
[Google dirve](https://drive.google.com/open?id=1ExrRSPVctHW8Nxrn0SofU1lVhK5Wn0_S), [Baidu Cloud: eg7b](https://pan.baidu.com/s/1KM5yV4IxHiAX4Iwn5G_TuA)

```Shell
# Training.
python train.py --config=yolact_resnet50_pascal_config
# Evalution.
python eval.py --trained_model=yolact_resnet50_pascal_112_120000.pth
```
