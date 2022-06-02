# PEPITA

Code to run the simulations of the paper:
### Error-driven Input Modulation: Solving the Credit Assignment Problem without a Backward Pass

Giorgia Dellaferrera, Gabriel Kreiman

https://arxiv.org/abs/2201.11665

To be presented at ICML 2022.


# Requirements
We run the experiments with the following:

Numpy framework (fully connected models): Python 3.9.5, Numpy 1.19.5, Keras 2.5.0

Pytorch framework (convolutional models): Python 3.7.10, Numpy 1.19.2, Pytorch 1.6.0


# Experiments  

## Fully connected models - numpy version

The experiments are run through `main.py`, which uses functions in `functions.py` and `utils.py`. 
The entire framework is numpy-based and relies on the keras library to load the datasets.

For example, to run PEPITA with the standard settings on the MNIST dataset:
```
python main.py --exp_name Experiment1 \
    --learn_type ERIN --n_runs 1 --train_epochs 100 \
    --sample_passes 2 --n_samples all --eta 0.1 --dropout 0.9 \
    --eta_decay --mnist --validation --batch_size 64 \
    --update_type mom --w_init he_uniform \
    --build auto --struct uniform --start_size 1024 --n_hlayers 1 --act_hidden relu --act_out softmax
``` 

Note that the training scheme for PEPITA is denoted as ERIN (ERror-INput). 
If you train with PEPITA (ERIN), make sure to use the setting `--sample_passes 2`, to have for each input two forward passes.

Substitute `--learn_type ERIN` with `--learn_type BP` to train the network with backpropagation. Remember to set `--sample_passes 1`.

## Convolutional models - Pytorch version

The experiments are run through `main_pytorch.py`, which uses functions in `models.py`. The entire framework is pytorch-based.

For example, to run PEPITA with the standard settings on the MNIST dataset:
```
python main_pytorch.py --exp_name Experiment2 \
    --learn_type ERIN --n_runs 1 --train_epochs 100 \
    --eta 0.01 --dropout 0.9 --Bstd 0.05 \
    --eta_decay --dataset mn --batch_size 50 \
    --update_type mom --w_init he_uniform \
    --model Net1conv1fcXL
``` 

The argument `Bstd` defines the standard deviation of the projection matrix. 
Here we use `B` instead of `F` (paper) to denote the projection matrix to avoid confusion with torch.nn.functional.

## Convergence rate

The notebook `plot_compute_slowness.py` contains the function to extract the convergence rate as "slowness" parameter.
