# Self-supervised feature extraction for ball detection in NAO robots
![SPQR TEAM](media/spqrteam.jpg)
This repository contains the implementation of a self-supervised learning approach for improving ball detection in NAO robots, specifically designed for the RoboCup Standard Platform League (SPL). The project addresses the challenge of training neural networks with limited labeled data by leveraging self-supervised learning techniques.

## Contributors

- Can Lin

- Marco Zimmatore

- Marco Volpini

- Penelope Malaspina

  
## Project Overview

Ball detection is a crucial component for NAO robots participating in RoboCup competitions. However, the traditional approach of manually labeling training data is resource-intensive and time-consuming. This project implements three different self-supervised learning approaches to enhance ball detection capabilities:

- **Triplet Loss**: Improves feature learning by comparing similar and dissimilar image patches
- **Colorization**: Trains the model to understand spatial and color relationships by reconstructing color from grayscale images
- **Edge Detection**: Enhances the model's ability to identify object boundaries and important visual features

The models are designed to work efficiently on NAO robots (NAO V5 & V4) with their hardware constraints (ATOM Z530 1.6 GHz CPU, 1 GB RAM).

## Key Features

- Self-supervised learning implementation for reduced dependency on labeled data
- Multiple pretext tasks for robust feature extraction
- Optimized for NAO robot hardware constraints
- Dataset creation and preprocessing tools
- Performance evaluation metrics and comparisons

## Repository Structure

```
.
├── docs/            # Documentation and research paper
├── results/         # Training results, model evaluations, and performance metrics
├── src/            # Source code for models implementations
├── utils/          # General utility functions and tools
├── weights/        # Saved model weights and checkpoints
├── main.ipynb      # Main notebook for model execution and testing
└── training.ipynb  # Training pipeline implementation and experiments
```

