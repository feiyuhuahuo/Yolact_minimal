## Yolact_minimal
Minimal PyTorch implementation of [Yolact:《YOLACT: Real-time Instance Segmentation》](https://arxiv.org/abs/1904.02689).  
The original project is [here](https://github.com/dbolya/yolact).  

This implementation simplified the original code, preserved the main function and made the network easy to understand.  
This implementation has not been updated to Yolact++.  

### The network structure.  
![Example 0](readme_imgs/network.png)

## Environments  
PyTorch >= 1.1  
Python >= 3.6  
onnxruntime-gpu == 1.6.0 for CUDA 10.2  
TensorRT == 7.2.3.4  
tensorboardX  
Other common packages.  

## Prepare
```Shell
# Build cython-nms 
python setup.py build_ext --inplace
```
- Download COCO 2017 datasets, modify `self.data_root` in 'res101_coco' in `config.py`. 
- Download weights.

Yolact trained weights.

|Backbone   | box mAP  | mask mAP | number of parameters | Google Drive                                                                                                             |Baidu Cloud                                                       |
|:---------:|:--------:|:--------:|:--------------------:|:------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
|Resnet50   | 31.5     | 29.3     |       31.16 M        |[todo]()    |[todo]() |
|Resnet101  | 32.9     | 30.5     |       50.15 M        |[todo]()    |[todo]() |
|swin_tiny  | 33.9     | 31.9     |       34.58 M        |[best_31.9_swin_tiny_coco_308000.pth](https://drive.google.com/file/d/12-RklMCIJ3nUsfP6veWa4s45_Q3s1tXD/view?usp=sharing) |[password: i8e9](https://pan.baidu.com/s/1laOjozNSwf2-mfFz87N_6A) |

ImageNet pre-trained weights.

| Backbone  | Google Drive                                                                                              |Baidu Cloud                                                        |
|:---------:|:---------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------:|
| Resnet50  | [backbone_res50.pth](https://drive.google.com/file/d/1Bx_DZbxVOgCgNVsEU3gH5p_rXKKxbVYo/view?usp=sharing)  | [password: juso](https://pan.baidu.com/s/12E5DEvb2zJYqa4T8fm08Og) |
| Resnet101 | [backbone_res101.pth](https://drive.google.com/file/d/1Q7Cj7j70a3nT6AEmsWXueq4VaaUQ7hwA/view?usp=sharing) | [password: 5wsp](https://pan.baidu.com/s/1ute7NHb2n3iDIiHxDOGwEg) |
| swin_tiny | [swin-tiny.pth](https://drive.google.com/file/d/1dvoPNGj2SHd5XhmSyE23GYzmimsTQbsU/view?usp=sharing)       | [password: g0o2](https://pan.baidu.com/s/1PTVtiryHXdDvuLcR4zJQFA) |

## Improvement log
2021.4.19. Use swin_tiny transformer as backbone, +1.0 box mAP, +1.4 mask mAP.  
2021.1.7. Focal loss did not help, tried conf_alpha 4, 6, 7, 8.  
2021.1.7. Less training iterations, 800k --> 680k with batch size 8.  
2020.11.2. Improved data augmentation, use rectangle anchors, training is stable, infinite loss no longer appears.  
2020.11.2. DDP training, train batch size increased to 16, +0.4 box mAP, +0.7 mask mAP (resnet101).  

## Train
```Shell
# Train with resnet101 backbone on one GPU with a batch size of 8 (default).
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --train_bs=8
# Train on multiple GPUs (i.e. two GPUs, 8 images per GPU).
export CUDA_VISIBLE_DEVICES=0,1  # Select the GPU to use.
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM)) train.py --train_bs=16
# Train with other configurations (res101_coco, res50_coco, res50_pascal, res101_custom, res50_custom, in total).
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --cfg=res50_coco
# Train with different batch_size (batch size should not be smaller than 4).
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --train_bs=4
# Train with different image size (anchor settings related to image size will be adjusted automatically).
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --img_size=400
# Resume training with a specified model.
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --resume=weights/latest_res101_coco_35000.pth
# Set evalution interval during training, set -1 to disable it.  
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --val_interval 8000
# Train on CPU.
python train.py --train_bs=4
```
## Use tensorboard
```Shell
tensorboard --logdir=tensorboard_log/res101_coco
```

## Evalution
```Shell
# Select the GPU to use.
export CUDA_VISIBLE_DEVICES=0
```

```Shell
# Evaluate on COCO val2017 (configuration will be parsed according to the model name).
# The metric API in this project can not get the exact COCO mAP, but the evaluation speed is fast. 
python eval.py --weight=weights/best_30.5_res101_coco_392000.pth
# To get the exact COCO mAP:
python eval.py --weight=weights/best_30.5_res101_coco_392000.pth --coco_api
# Evaluate with a specified number of images.
python eval.py --weight=weights/best_30.5_res101_coco_392000.pth --val_num=1000
# Evaluate with traditional nms.
python eval.py --weight=weights/best_30.5_res101_coco_392000.pth --traditional_nms
```
## Detect
- detect result  
![Example 2](readme_imgs/result.jpg)  
  
```Shell
# Select the GPU to use.
export CUDA_VISIBLE_DEVICES=0
```

```Shell
# To detect images, pass the path of the image folder, detected images will be saved in `results/images`.
python detect.py --weight=weights/best_30.5_res101_coco_392000.pth --image=images
```
- cutout object  
![Example 3](readme_imgs/cutout.jpg)
```Shell
# Use --cutout to cut out detected objects.
python detect.py --weight=weights/best_30.5_res101_coco_392000.pth --image=images --cutout
```
```Shell
# To detect videos, pass the path of video, detected video will be saved in `results/videos`:
python detect.py --weight=weights/best_30.5_res101_coco_392000.pth --video=videos/1.mp4
# Use --real_time to detect real-timely.
python detect.py --weight=weights/best_30.5_res101_coco_392000.pth --video=videos/1.mp4 --real_time
```
- linear combination result  
![Example 4](readme_imgs/lincomb.jpg)

```Shell
# Use --hide_mask, --hide_score, --save_lincomb, --no_crop and so on to get different results.
python detect.py --weight=weights/best_30.5_res101_coco_392000.pth --image=images --save_lincomb
```

## Transport to ONNX    
```Shell
python export2onnx.py --weight='weights/best_30.5_res101_coco_392000.pth' --opset=12
# Detect with ONNX file, all the options are the same as those in `detect.py`.
python detect_with_onnx.py --weight='onnx_files/res101_coco.onnx' --image=images.
```

## Accelerate with TensorRT   
```Shell
python export2trt.py --weight='onnx_files/res101_coco.onnx'
# Detect with TensorRT, all the options are the same as those in `detect.py`.
python detect_with_trt.py --weight='trt_files/res101_coco.trt' --image=images.
```

## Train on PASCAL_SBD datasets
- Download PASCAL_SBD datasets from [here](http://home.bharathh.info/pubs/codes/SBD/download.html), modify the path of the `img` folder in `data/config.py`.  
```Shell
# Generate a coco-style json.
python utils/pascal2coco.py --folder_path=/home/feiyu/Data/pascal_sbd
# Training.
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --cfg=res50_pascal
```

## Train custom datasets
- Install labelme  
```Shell
pip install labelme
```
- Use labelme to label your images, only ploygons are needed. The created json files are in the same folder with the images.  
![Example 5](readme_imgs/labelme.png)
- Prepare a 'labels.txt' like this, this first line: 'background' is always needed.  
![Example 6](readme_imgs/labels.png)
- Prepare coco-style json, pass the paths of your image folder and the labels.txt. The 'custom_dataset' folder is a prepared example.  
```Shell
python utils/labelme2coco.py --img_dir=custom_dataset --label_name=cuatom_dataset/labels.txt
```
- Edit `CUSTOM_CLASSES` in `config.py`.  
![Example 7](readme_imgs/label_name.png)  
Note that if there's only one class, the `CUSTOM_CLASSES` should be like `('dog', )`. The final comma is necessary to make it as a tuple, or the number of classes would be `len('dog')`.  
- Choose a configuration ('res101_custom' or 'res50_custom') in `config.py`, modify the corresponding `self.train_imgs` and `self.train_ann`. If you need to validate, prepare the validation dataset by the same way.  
- Then train.  
```Shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM)) train.py --cfg=res101_custom
```
- Some parameters need to be taken care of by yourself:
1) Training batch size, try not to use batch size smaller than 4.
2) Anchor size, the anchor size should match with the object scale of your dataset.
3) Total training steps, learning rate decay steps and the warm up step, these should be decided according to the dataset size, overwrite `self.lr_steps`, `self.warmup_until` in your configuration.

## Citation
```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```