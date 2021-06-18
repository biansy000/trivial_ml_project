# The code for ML class


## Brief introduction
In the project, we need to find the boundary between different cells. The data set consists of 30 samples, 25 for train and 5 for test given by the teacher. No pretrain or extra data is allowed. 

In the repo, we implement three methods, namely:
* UNet
* UNet++
* HRNet


## Run the code
First, you need to put data in the ``data`` directory.

Then you can use the following lines to run the given methods. Examples are given using the default arguments.

**UNet**:
```
python train_unet.py
```

**UNet++**:
```
python train_plusplus.py
```

**HRNet**:
```
python train_hrnet.py
```

<!-- ### Code comment
We think that the original repo is quite easy to understand for a well-qualiified student who is interested in AI, thus forgive us for not too much comment in the code. -->


### Some other words
The repo is originally forked from [Pytorch-Unet](https://github.com/milesial/Pytorch-UNet), but we have changed most part of it. 
<!-- The reason that we do not directly forked it on github is that the github only supports public fork, which means that everyone can see our work and is not so suitable for a class assignment. -->
For convenience and potiential needs, we have kept the original ``README.md`` file, and rename it as ``README_old.md``.

