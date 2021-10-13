# S<sup>3</sup>ADNet: Sequential Anomaly Detection with Pessimistic Contrastive Learning
Self-supervised Sequential Anomaly Detection Network

## 1. Setup the environment (using [Miniforge](https://github.com/conda-forge/miniforge))
   
```shell
$ git clone https://github.com/S3ADNet/S3ADNet.git
$ cd S3ADNet
$ conda env create -f environment.yml
$ conda activate s3adnet
```


## 2. Run experiments

KDDCup99
```shell
$ python main.py \
 --experiment kddcup99 \
 --embedding-noise \
 --seq_len 16 \
 --batch_size 256 \
 --max_epochs 50
```

HASC
```shell
$ python main.py \
 --experiment hasc \
 --feature-drop 0.1 \
 --win_size 100 \
 --seq_len 4 \
 --batch_size 1 \
 --max_epochs 50
```
