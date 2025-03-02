# DPDNet
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)    

DPDNet：The Lightweight Stereo Matching Network Based on Disparity Probability Distribution Consistency.  

**Our network pre-trained weight files are stored in [DPDNet](https://pan.baidu.com/s/1PVhhc3Kxnf09P9Hccj0SzA?pwd=nsdg).**    
This including (2D_DPDNet_SF.  2D_DPDNet_SF_DS.  2D_DPDNet_SF_DS_12+15.  2D_DPDNet_SF_DS_12+15_12.  2D_DPDNet_SF_DS_12+15_15.  3D_DPDNet_SF.  3D_DPDNet_SF_DS.  3D_DPDNet_SF_DS_12+15.  3D_DPDNet_SF_DS_12+15_12.  3D_DPDNet_SF_DS_12+15_15.) 

### Requirements
The code is tested on:
- Ubuntu 18.04
- Python 3.9
- PyTorch 1.12.1 
- Torchvision 0.13.1
- CUDA 11.3
## Evaluation Results 
DPDNets are trained and tested using [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (SF), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_stereo.php) and [DrivingStereo](https://drivingstereo-dataset.github.io/) (DS) datasets. 

### Calculate operations and parameters. 
Run the following command to see the complexity in terms of number of operations and parameters.   
```shell
python cost.py
```

### Training 

Set a variable for the dataset directory, e.g. ```DATAPATH="/Datasets/SceneFlow/"```. Then, run ```train.py``` as below:

#### Pretraining on SceneFlow
```shell
python train.py --dataset sceneflow --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt --epochs 20 --lrepochs "10,12,14,16:2" --batch_size 4 --test_batch_size 4 --model DPDNet2D --logdir="save_file"
```

#### Finetuning on DrivingStereo

```shell
python train.py --dataset drivingstereo --datapath $DATAPATH --trainlist ./filenames/driving_test.txt --testlist ./filenames/driving_train.txt --epochs 20 --lrepochs "10,12,14,16:2" --batch_size 4 --test_batch_size 4 --model DPDNet2D --logdir="save_file"
```

#### Finetuning on KITTI

```shell
python train.py --dataset kitti --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt --epochs 400 --lrepochs "200:10" --batch_size 4 --test_batch_size 4 --loadckpt ./checkpoints/pretrained.ckpt --model DPDNet2D --logdir="save_file"
```

The arguments in both cases can be set differently depending on the model, dataset and hardware resources.

### Prediction

The following script creates disparity maps for a specified model:

```shell
python prediction.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --loadckpt ./checkpoints/finetuned.ckpt --dataset kitti --colored True --model DPDNet2D
```

## Credits

The implementation of this code is based on [MobileStereoNet](https://github.com/cogsys-tuebingen/mobilestereonet).
