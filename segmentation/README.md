# Applying EfficientMod to Semantic Segmentation

Our semantic segmentation implementation is based on [MMSegmentation v0.19.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.19.0) and [PVT segmentation](https://github.com/whai362/PVT/tree/v2/segmentation). Thank the authors for their wonderful works.

## Usage

Install MMSegmentation v0.19.0. 

## Data preparation

Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets) in MMSegmentation.


## Results and models

| Method | Backbone | Pretrain | Iters | mIoU  | Download |
| --- | --- | --- |:---:|:---:| --- |
| Semantic FPN | EfficientMod-s-Conv  | ImageNet-1K |  40K  |     43.5   | [[checkpoint & log]](https://drive.google.com/drive/folders/1XXTCgh4o5sNrSdGmuqqPK22TvmRsaUvk?usp=share_link) |
| Semantic FPN | EfficientMod-s  | ImageNet-1K |  40K  |     46.0   |[[checkpoint & log]](https://drive.google.com/drive/folders/1ih0zO9X1yklbsVOHSEIeNn5goleCkFxs?usp=share_link) |


## Evaluation
To evaluate EfficientMod + Semantic FPN on a single node with 8 GPUs run:
```
dist_test.sh configs/sem_fpn/{configure-file}.py /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```


## Training
To train EfficientMod + Semantic FPN on a single node with 8 GPUs run:

```
dist_train.sh configs/sem_fpn/{configure-file}.py 8
```
