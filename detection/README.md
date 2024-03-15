# Applying EfficientMod to Object Detection

Our detection implementation is based on [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0) and [PVT detection](https://github.com/whai362/PVT/tree/v2/detection). Thank the authors for their wonderful works.



## Usage

Install [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0) from souce cocde,

or

```
pip install mmdet==2.19.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```


## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0).


## Results and models on COCO


| Method     | Backbone | Pretrain    | Lr schd  | box AP | mask AP | Download |
|------------|----------|-------------|:-------:|:---:|:------:|-----|
| Mask R-CNN  | EfficientMod-s-Conv | ImageNet-1K |    1x   | 42.1  |  38.5  |   [checkpoint & log](https://drive.google.com/drive/folders/1EYplGBr0osoITnYlA_ImbSGgfWBFbBuf?usp=share_link) |
| Mask R-CNN   | EfficientMod-s | ImageNet-1K |    1x   | 43.6  |  40.3  |  [checkpoint & log](https://drive.google.com/drive/folders/1hiZst1cbvYiIFJ6dnPb4KUOfxq3mjzha?usp=share_link) |



## Evaluation

To evaluate EfficientMod + Mask R-CNN on COCO val2017, run:
```
dist_test.sh configs/{configure-file} /path/to/checkpoint_file 8 --out results.pkl --eval bbox segm
```


## Training
To train EfficientMod + Mask R-CNN on COCO train2017:
```
dist_train.sh configs/{configure-file} 8
```
