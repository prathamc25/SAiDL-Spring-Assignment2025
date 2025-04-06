
# Robust Loss Functions for Noisy Label Learning 

[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains a PyTorch implementation for training neural networks on datasets with noisy labels. The code evaluates various loss functions designed to be robust against label noise.

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Loss Functions](#loss-functions)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Noise Types](#noise-types)
- [Experiments](#experiments)
- [Running the Code](#running-the-code)
- [Visualization](#visualization)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)

## Overview

Training deep neural networks requires large amounts of labeled data, but the process of collecting accurate labels can be challenging. This project explores different loss functions designed to handle noisy labels effectively, improving model robustness in real-world scenarios.

Take a look at our [noise pattern visualizations](./Noisy%20dataset%20Plots/Noise_patterns.pdf) to understand the types of label corruption we're addressing.

## Features

- Implementation of multiple loss functions for dealing with noisy labels
- Support for both symmetric and asymmetric noise
- Configurable noise rates
- Complete training pipeline for CIFAR-10 dataset
- Visualization tools for result analysis

## Loss Functions

The implemented loss functions are organized into three groups:

### Vanilla Losses
- **CE**: Standard Cross-Entropy
- **MAE**: Mean Absolute Error 
- **RCE**: Reverse Cross-Entropy

### Normalized Losses
- **NCE**: Normalized Cross-Entropy
- **NRCE**: Normalized Reverse Cross-Entropy
- **NFL**: Normalized Focal Loss

### APL (Active Passive Losses)
- **NCE+RCE**: Normalized Cross-Entropy + Reverse Cross-Entropy
- **NCE+MAE**: Normalized Cross-Entropy + Mean Absolute Error
- **NFL+RCE**: Normalized Focal Loss + Reverse Cross-Entropy
- **NFL+MAE**: Normalized Focal Loss + Mean Absolute Error
- **NFL+NCE**: Normalized Focal Loss + Normalized Cross-Entropy
- **MAE+RCE**: Mean Absolute Error + Reverse Cross-Entropy

For implementation details, check out the [CoreML_Implementation.ipynb](./CoreML_Implementation.ipynb) notebook.

## Model Architecture

The project uses a CNN8 architecture:
- 8-layer CNN with BatchNorm
- 3 blocks of ConvBN-ConvBN-MaxPool
- 256 feature maps in the last convolutional layer
- 2-layer fully connected classifier

The complete model implementation can be found in the [implementation notebook](./CoreML_Implementation.ipynb).

## Configuration

The main configuration parameters include:

```python
class Config:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "./data"
    noise_rate = 0.6  # Adjustable noise rate
    asym = False      # Toggle for asymmetric noise
    num_classes = 10
    batch_size = 128
    epochs = 200
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    loss_type = "NCE+RCE"  # Default loss function
    alpha = 1.0  # Weighting for first loss component
    beta = 1.0   # Weighting for second loss component
```

## Noise Types

The code supports two types of label noise:

- **Symmetric Noise**: Labels are randomly flipped to any other class with equal probability
- **Asymmetric Noise**: Labels are flipped to specific classes based on a pre-defined transition matrix (mimicking real-world confusion patterns)

You can see visualizations of these noise patterns in the [Noisy dataset Plots](./Noisy%20dataset%20Plots/) directory.

## Experiments

The experiment suite evaluates all loss functions against varying noise levels:

- **Symmetric Noise**: 20%, 40%, 60%, 80%
- **Asymmetric Noise**: 10%, 20%, 30%, 40%

For detailed results and analysis, see the [implementation notebook](./CoreML_Implementation.ipynb).

## Running the Code

The complete training pipeline is provided in the [implementation notebook](./CoreML_Implementation.ipynb). To run an experiment:

1. Set the desired configuration parameters
2. Select the noise type and rate
3. Choose the loss function
4. Execute the training function

```python
config = Config()
config.noise_rate = 0.4
config.asym = False  # Symmetric noise
config.loss_type = "NCE+RCE"
results = train_model(config)
```

## Visualization

The code includes functions to visualize:
- Training/test loss curves
- Training/test accuracy curves
- Comparison plots across noise rates
- Group comparisons between different loss function families

Example visualizations can be found in the [Noisy dataset Plots](./Noisy%20dataset%20Plots/) directory.

## Directory Structure

```
CoreML/
├── CoreML_Implementation.ipynb   # Main implementation notebook
├── Noisy dataset Plots/          # Visualization outputs
│   └── Noise_patterns.pdf        # Visualization of noise patterns
└── readme.md                     # This documentation file
```

## Requirements

- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

Install dependencies with:
```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Citation

If you find this code useful in your research, please consider citing the original papers that introduced these loss functions:

```bibtex
@inproceedings{ma2020normalized,
  title={Normalized Loss Functions for Deep Learning with Noisy Labels},
  author={Ma, Xingjun and Huang, Hanxun and Wang, Yisen and Romano, Simone and Erfani, Sarah and Bailey, James},
  booktitle={ICML},
  year={2020}
}
```

## License

This project is open-source and available for research and educational purposes.
```
