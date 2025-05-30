# SubGEC: Subgraph Gaussian Embedding Contrast for Self-Supervised Graph Representation Learning

This repository contains the implementation of **Subgraph Gaussian Embedding Contrast (SubGEC)**, a method for self-supervised node representation learning, as described in our paper accepted to **ECML-PKDD 2025**.

* 📄 [Paper on arXiv](http://arxiv.org/abs/2505.23529)

## Overview

Graph Representation Learning (GRL) aims to encode high-dimensional graph-structured data into low-dimensional vectors. Self-supervised learning (SSL) methods, particularly contrastive learning, leverage data similarities and differences without extensive human annotations.

We propose **SubGEC**, introducing a novel **Subgraph Gaussian Embedding (SGE)** module. This module adaptively maps subgraphs to a structured Gaussian space, ensuring the preservation of input subgraph characteristics while controlling the distribution through Gaussian regularization. We utilize optimal transport distances—specifically the **Wasserstein** and **Gromov-Wasserstein** distances—to robustly measure subgraph similarity for contrastive learning. Extensive experiments demonstrate that SubGEC achieves competitive or superior performance compared to state-of-the-art approaches.

## Authors

* **Shifeng XIE** \[[LinkedIn](https://www.linkedin.com/in/shifeng-xie-953757209)]
* **Aref Einizade** \[[Google Scholar](https://scholar.google.com/citations?user=ZEQyAaAAAAAJ&hl=en)]
* **Jhony H. Giraldo** \[[Google Scholar](https://scholar.google.com/citations?user=iwzmGKwAAAAJ&hl=en)]

---

## Usage

### Training with Random Hyperparameter Search

To start training the model with random hyperparameter search using [Optuna](https://optuna.org/), run:

```bash
python train.py
```

This script performs hyperparameter optimization over a predefined search space. Adjust the search space and the number of trials in the `train.py` script.

### Training with Specific Hyperparameters

If you prefer to specify certain hyperparameters, use:

```bash
python singleTrain.py
```

---

## Model Architecture

The SubGEC model architecture includes:

* **Graph Encoder**: Two Graph Convolutional Network (GCN) layers that produce initial node embeddings.
* **Subgraph Sampling**: Breadth-First Search (BFS) extracts subgraphs around selected anchor nodes.
* **Subgraph Gaussian Embedding (SGE) Module**: Composed of GraphSAGE followed by two separate Graph Attention Networks (GATs) to encode the means and variances of subgraph embeddings into Gaussian distributions.
* **Gaussian Regularization**: Uses Kullback–Leibler (KL) divergence to enforce Gaussian-distributed embeddings.
* **Contrastive Learning**: Employs the Wasserstein and Gromov-Wasserstein distances in an optimal transport framework to measure subgraph similarity.

---
