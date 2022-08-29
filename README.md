# SiamMSS

## 1. Environment setup
This code has been tested on windows 10, Python 3.7.9, Pytorch 1.7.0, CUDA 10.0.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test

Download the pretrained model and put them into `tools/snapshot` directory.   
From BaiduYun:
* [SiamMSS](https://pan.baidu.com/s/1luhtEuuusjme9BzL1CR-ng) extract code: k9vh  
* [our_train_baseline](https://pan.baidu.com/s/1DCJGMvT26oqHZKtYXzzYkg) extract code: gcvw  
* [baseline](https://github.com/ohhhyeahhh/SiamGAT) 



 If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

The tracking result can be download from [BaiduYun](https://pan.baidu.com/s/1-iZUcZhqzERt_0DCXwkzEw) (extract code: 50fq) for comparision.

```bash 
python testTracker.py \    
        --config ../experiments/siamsmm_googlenet/config.yaml \
	--dataset LSOTB \                                 # dataset_name
	--snapshot snapshot/siamsmm.pth              # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 3. Train

### Prepare training datasets

Download the datasets：
* [LSOTB](https://github.com/QiaoLiuHit/LSOTB-TIR)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)


**Note:** `training_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.

### Download pretrained backbones
Download pretrained backbones from [link](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth) and put them into `pretrained_models` directory.

### Train a model
To train the SiamSMM model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
```


## 5. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot) and [SiamGAT](https://github.com/ohhhyeahhh/SiamGAT). We would like to express our sincere thanks to the contributors.



