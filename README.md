# SubGEC: Subgraph Gaussian Embedding Contrast for Self-Supervised Representation Learning

This repository contains the implementation of **Subgraph Gaussian Embedding Contrast (SubGEC)**, a method for self-supervised node representation learning, as described in our paper:


## Overview

Graph representation learning (GRL) is fundamental for encoding graph-structured data into low-dimensional vectors. Self-supervised learning (SSL) methods, especially contrastive learning, have shown promise in GRL by leveraging similarities and differences between data samples without requiring extensive human annotation.

In this work, we propose **SubGEC**, which introduces a **Subgraph Gaussian Embedding (SGE)** module. This module adaptively maps subgraphs to a structured Gaussian space, ensuring the preservation of graph characteristics while controlling the distribution of generated subgraphs. We employ optimal transport distances, including **Wasserstein** and **Gromov-Wasserstein** distances, to effectively measure the similarity between subgraphs, enhancing the robustness of the contrastive learning process.

Our method outperforms or presents competitive performance against state-of-the-art approaches on multiple benchmarks.

---


## Usage

### Training with Random Hyperparameter Search

To start training the model with random hyperparameter search using [Optuna](https://optuna.org/), run:

```bash
python train.py
```

This script performs hyperparameter optimization over a predefined search space. You can adjust the search space and the number of trials in the `train.py` script.

### Training with Specific Hyperparameters

If you prefer to specify certain hyperparameters, use:

```bash
python singleTrain.py 
```
---

## Model Architecture

The SGEC model consists of:

- **Graph Encoder**: Two Graph Convolutional Network (GCN) layers for initial graph embedding.
- **Subgraph Sampling**: Breadth-First Search (BFS) is used to sample subgraphs centered around selected nodes.
- **Subgraph Gaussian Embedding (SGE) Module**: Combines GraphSAGE and Graph Attention Network (GAT) layers to embed subgraphs into a Gaussian space.
- **Contrastive Learning with Optimal Transport**: Utilizes Wasserstein and Gromov-Wasserstein distances to measure similarities between subgraphs for contrastive learning.

---

