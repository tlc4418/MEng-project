# MEng Individual Project

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo contains the codebase for Thomas Coste's MEng Individual Project on 'Object-Centric Respresentations for Cognitive Reinforcement Learning Tasks,' supervised by Prof. Murray Shanahan. This work was conducted as part of the MEng Degree in Computing (Machine Learning and Artificial Intelligence) at Imperial College London.


## Table of Contents
- [Animal-AI Environment](#animal-ai-environment)
- [Segmentation Visualizations](#segmentation-visualizations)
- [Data Generation](#data-generation)
  - [Arena Creator](#arena-creator)
  - [AAI Dataset Generator](#aai-dataset-generator)
- [Reinforcement Learning](#reinforcement-learning)
    - [Agent Types](#agent-types)
    - [Visualization](#visualization)
- [Slot Attention and AlignNet](#slot-attention-and-alignnet)
    - [Training and Testing](slot_attention_and_alignnet/README.md#training-and-testing)
    - [Data Processing](slot_attention_and_alignnet/README.md#data-processing)
    - [Model Implementations](slot_attention_and_alignnet/README.md#model-implementations)
    - [Plotting](slot_attention_and_alignnet/README.md#plotting)
    - [Utils](slot_attention_and_alignnet/README.md#utils)


## Animal-AI Environment
[aai_environment/](aai_environment/) is a placeholder for the Animal-AI (AAI) Environment from https://github.com/mdcrosby/animal-ai. 


## Segmentation Visualizations
[aai_segmentation/](aai_segmentation/) contains helpful visualizers for ground-truth mask segmentations. The a color range helper is used to find appropriate color ranges for creating ground truth masks. The ARI visualizer can help illustrate mask comparisons before ARI calculation and see the corresponding ARI score.


## Data Generation
[dataset_generation/](dataset_generation/) is used to generate datasets used throughout the project.

### Arena Creator
The arena creator creates Animal-AI arenas with a customizable range of object types and count. Object placement and even size are random at each environment reset through configuration randomization.

### AAI Dataset Generator
The AAI dataset generator collects images from a group of either competition or custom arenas. It resets environments iteratively, collecting the first image observed if it meets certain customizable conditions such as object distance and count. 


## Reinforcement Learning
[rl_agents/](rl_agents/) holds all files linked to reinforcement learning agents. Most important is the training script which trains a Stable Baselines 3 DQN agent on the AAI environment. Parameters are easily customizable and training curves are plotted to Tensorboard. There are multiple types of agents possible, depending on custom feature extractor classes.

### Agent Types
The feature extractors can be one of the following:
- The **base, or default, extractor**. This is a standard CNN.
- A **Slot Attention extractor**. This uses a pre-trained Slot Attention model as the feature extractor and concatenates the resulting slot representations.
- An **AlignNet extractor**. This uses a pre-trained SA+AlignNet model as the feature extractor. This also requires stacking environment frames into time sequences, in order to be compatible with AlignNet.

### Visualization
A trained agent's behaviour can be visualized with the trained agent visualizer. This runs an AAI instance on an a given task and uses the trained model to predict actions. These actions are executed and the resulting observations are saved and printed out at the end.

## Slot Attention and AlignNet
[slot_attention_and_alignnet/](slot_attention_and_alignnet/) contains a large number of files relating to the Slot Attention and AlignNet models and experiments. A [separate README](slot_attention_and_alignnet/README.md) is dedicated to this folder.

