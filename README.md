# Siamese Neural Network for Record Linkage

This repository implements a Siamese Neural Network (SNN) as a classifier for Record Linkage. This task was completed as part of the **Innohack** hackathon. The project includes the main functions for the SNN, with additional components such as graph processing, blocking, etc., available in the [unfinished project repository](https://gitflic.ru/project/task-force-141/innohack-solution).

## Overview

The SNN consists of two subnetworks, each representing a `transformerEncoder`. These subnetworks convert records into vectors, and the SNN optimizes `ContrastiveLoss` to maximizes or minimizes the distance between these vectors depending on the label.

## Key Features

- **Dual Subnetworks**: Each subnetwork is a `transformerEncoder` that processes records into vectors.
- **Contrastive Loss**: The model optimizes ContrastiveLoss to adjust the distance between vectors based on the label.
- **Record Linkage**: The SNN is designed to classify records based on their similarity, making it suitable for Record Linkage tasks.

## Potential Improvements

While there are many ways to potentially enhance the SNN, such as changing the optimizer, tuning hyperparameters, or altering the mean sequence, these are not the focus of this README. Instead, we will discuss a more critical aspect of the model.

### Attribute-Aware Record Processing

Currently, the SNN treats each record as a single entity, ignoring the information about its individual attributes. This approach was chosen under the assumption that the model would still learn all necessary dependencies, albeit with more data, memory, and slower training times.

To address this, we propose the following:

1. **Measure of differences**: Choose a measure of differences (e.g., Euclidean distance) for attribute vectors. Different measures can be selected for different attributes.
2. **Fully Connected Neural Network (FCNN)**: Create an FCNN that takes as input a tensor representing the differences in attributes and output predicted label.
3. **Non-linear Model**: Avoid using a simple linear model (or weighted sum of attributes) due to the uncertainty about linear separability.

## License

This project is licensed under the [MIT License]. Check out LICENSE.txt for the full text.
