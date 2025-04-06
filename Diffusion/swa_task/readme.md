
# Sliding Window Attention (SWA) for Diffusion Models

## Overview

This project implements and evaluates Sliding Window Attention mechanisms in diffusion models. Sliding Window Attention is an efficient attention variant that restricts the receptive field of self-attention to a local window around each token, making it computationally efficient for processing long sequences while maintaining good performance.

## Implementation Details

The implementation is contained in the Jupyter notebook `SWA_task.ipynb`, which includes:

- Model architecture with Sliding Window Attention
- Training pipeline for diffusion models
- Evaluation metrics and visualization
- Hyperparameter configurations

## Key Features

- Efficient attention mechanism for diffusion models
- Reduced computational complexity compared to full attention
- Scalable approach for generating high-quality images
- Compatible with various diffusion model architectures

## Results

After training for 60 epochs, the model demonstrates:
- Stable convergence with consistent sample quality
- Effective learning of image distributions
- Computational efficiency compared to baseline models
- [View detailed results in the notebook](SWA_task.ipynb)`  
- [View detailed results of the fid/cmmd/siglip/align tasks](evaluation_results.json)

## Model Checkpoints

Trained model checkpoints (after 60 epochs) are available on Google Drive:

[**Download Model Checkpoints**](https://drive.google.com/drive/folders/19615y6WDKlJyn0svd8NbTt_jTy_NzPKs)

## Usage

To use the pre-trained model:

1. Download the checkpoints from the Google Drive link
2. Place the checkpoint files in a directory accessible to your code
3. Load the model using:

```python
from model import SlidingWindowDiffusionModel  # Adjust import based on actual implementation

# Load the model
model = SlidingWindowDiffusionModel.load_from_checkpoint("path/to/checkpoint")

# Generate samples
samples = model.sample(batch_size=4)
```

## Dependencies

- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- diffusers (optional, depending on implementation)



## Acknowledgements

This work was completed as part of the SAiDL Spring Assignment 2025.
```
