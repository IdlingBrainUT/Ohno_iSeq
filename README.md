# iSeq (Ohno et al. from Inokuchi Lab.)

## Overview

iSeq (or its GPU implementation, `igpu`) automatically extracts 'neural sequences,' which are patterns of neuronal population activity with specific temporal order relationships, from raster data with neuron × time dimensions. This program is specifically tuned for analyzing data obtained using calcium imaging techniques.

For details on the methodology and examples of its applications, please refer to the [preprint](https://doi.org/10.1101/2025.01.16.633469).

## System requirements

### Hardware requirements

A GPU must be available and controllable via `CUDA`.

### Software requirements

OS: Windows, Linux

These codes has been tested in the following systems:

 - OS: Windows 11 Pro
 - GPU: NVIDIA GeForce RTX 4080 SUPER

Additionally, Python must be installed, and `pip` should be available for downloading the required packages.

## Installation guide

### Instructions

Place `igpu` in the project folder and install the required Python libraries according to the error messages.

### Typical install time

There is no need to install `igpu` itself, so you can start using it immediately.

## Demo

### Instructions to run on data

For a demo on using `igpu`, refer to the Jupyter Notebook in `How_to_use_iqpu.ipynb` and execute it sequentially from top to bottom.

### Expected output

It analyzes synthetic data embedded with three neural sequences and should correctly detect approximately three sequences. Additionally, an example code for beautifully visualizing the obtained results is provided.

### Expected run time

This demo should run within a few minutes on a typical desktop computer.

## Instructions for use

### How to run the software on your data

First, refer to the demo (`How_to_use_iqpu.ipynb`).

Suppose your data matrix, `x`, is loaded, where each row records the temporal changes in activity for each neuron. Running the following code will initiate matrix decomposition, and the results will be stored in model. The shapes of the detected neural sequences are stored in `model.W`, while the temporal changes in their activity intensities are stored in `model.H`. 

```python
import igpu

model = igpu.solve(x, k=20, l=50, z_th=x.min(), random_seed=0, corr_max=0.3, comp_rate=0.1)
```

The options that can be specified in `igpu.solve` are as follows:

 - `V` (required): The matrix to be decomposed.
 - `k` (required): The maximum number of neural sequences to detect.
 - `l` (required): The maximum duration of the neural sequences.
 - `z_th`: Minimum threshold (values below this are considered zero, default=`0.001`).
 - `tolerance`: The tolerance for computation (default=`1e-7`).
 - `n_iter`: The number of computation iterations per phase ([searching for high intensity columns, decomposing the high intensity matrix, adjustment after determining the number of neural sequences, final decomposition], default=`[30, 30, 10, 30]`).
 - `comp_rate`: The compression ratio when reducing the matrix V (default=`0.3`).
 - `Wlim`: The maximum value for matrix W (default=`0.1`).
 - `Hlim`: The maximum value for matrix H (default=`1.0`).
 - `corr_max`: The correlation threshold for considering two neural sequences as identical (default=`0.95`).
 - `random_seed`: The seed value for random number generation (default=`None`).

For details on each parameter, please refer to the paper.

### Reproduction instruments

For a more practical application of `igpu` that we used (repeating igpu decomposition and selecting the most accurate results), please refer to the notebook as follows:

 - `synthetic_data_decomposition.ipynb`
 - `decompose_matrix_by_iseq.ipynb`

Additionally, other notebooks contain code for visualizing the results obtained with `igpu`. These codes are the same ones used to generate the figures in our paper, allowing you to reproduce our results.