# [Efficient Modulation for Vision Networks] (ICLR 2024)

## News & TODO & Updates:
-  [ ] will improve the performance with better training recipe.
-  [ ] Upload benchmark script to ease latency benchmark.

## Image Classification
### 1. Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; timm==0.6.13;  

data prepare: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```



### 2. Pre-trained Context Cluster Models
We upload the **checkpoints** with distillation and **logs** to google drive. Feel free to download.

| Model    |  #params | Image resolution | Top1 Acc|  Download | 
| :---     |   :---:    |  :---: |  :---:   |:---:  |
| EfficientMod-xxs  |   4.7M     |   224 |  77.1 |  [[checkpoint & logs]](https://drive.google.com/drive/folders/1c0dlnN7w1bHlAsKcJFhGVA2mIhoA6ZHz?usp=sharing) |
| EfficientMod-xs |   6.6M     |   224 |  79.4  | [[checkpoint & logs]](https://drive.google.com/drive/folders/1PPQFO891WfJRUiH58NlWOgHDEzDnwC0_?usp=share_link) |
| EfficientMod-s |   12.9M     |   224 |  81.9  | [[checkpoint & logs]](https://drive.google.com/drive/folders/1rJs8LcWmdTFmj-IJ0cmlVp_MxGfZFsFk?usp=share_link) |
| EfficientMod-s-Conv (No Distill.) |   12.9M     |   224 |  80.5  | [[checkpoint & logs]](https://drive.google.com/drive/folders/1EY637XRiDPL4AwrVGESJWsK-ZP2GhnaI?usp=share_link) |

### 3. Validation

To evaluate our EfficientMod models, run:

```bash
python3 validate.py /path/to/imagenet  --model {model} -b 256 --checkpoint {/path/to/checkpoint} 
```



### 4. Train
We show how to train EfficientMod on 8 GPUs.

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --data {path-to-imagenet} --model {model} -b 256 --lr 4e-3 --amp --model-ema --distillation-type soft --distillation-tau 1 --auto-resume --exp_tag {experiment_tag}

```



**See folder [detection](detection/) for Detection and instance segmentation tasks on COCO.**.

**See folder [segmentation](segmentation/) for Semantic Segmentation task on ADE20K.**

## BibTeX

    @inproceedings{
        ma2024efficient,
        title={Efficient Modulation for Vision Networks},
        author={Xu Ma and Xiyang Dai and Jianwei Yang and Bin Xiao and Yinpeng Chen and Yun Fu and Lu Yuan},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=ip5LHJs6QX}
    }
