# FGMCD

Source code of **Understanding Brand’s Association: Fine-Grained Multi-Modal Content Discovery For Brands**

**Keywords:** Brand Association, Multi-Modal, Content Discovery

## Requirements

#### Environments

* **Ubuntu** 18.04
* **CUDA** 11.2
* **Python** 3.7
* **PyTorch** 1.7.0

We used virtualenv to setup a deep learning workspace that supports PyTorch.
Run the following script to install the required packages.

#### Required Data

Original data can be accessed through the specific server from HITSZ.
Contact relevant person for more information.

## Getting started

Run the following script to preprocess the original data.

Make sure the **source** path and **target** path are right.

**Remember**

- original data -> `/root/brand/ins_Car_data`
- your working directory -> `/root/$YOUR_USER_NAME$/FGMCD`
- feature data -> `/root/$YOUR_USER_NAME$/insCar`

```shell
conda activate base
pip install -r requirements.txt
python preprocess/preprocess.py
```

## Training

Run the following script to train the model.

```shell
sh bin/A100.sh # GPU enabled
sh bin/public_cluster # CPU only
```

Remember set right path of `ROOT_PATH` and `Python` path.

## Dataset Structure

Store the training, validation and test subset into three folders in the following structure respectively.

```shell
${subset_name}
├── FeatureData
│   └── ${feature_name}
│       ├── feature.bin
│       ├── shape.txt
│       └── id.txt
├── ImageSets
│   └── ${subset_name}.txt
└── TextData
    └── ${subset_name}.caption.txt

```

* `FeatureData`: video frame features.
  Using [txt2bin.py](https://github.com/danieljf24/simpleknn/blob/master/txt2bin.py) to convert video frame feature in
  the required binary format.
* `${subset_name}.txt`: all video IDs in the specific subset, one video ID per line.
* `${dsubset_name}.caption.txt`: caption data. The file structure is as follows, in which the video and sent in the same
  line are relevant.
