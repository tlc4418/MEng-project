# Slot Attention and AlignNet 

## Training and Testing
This folder contains multiple training and testing files for both the Slot Attention and SA+AlignNet models. These are used to train models on specified datasets, save intermediate models, visualize intermediate representation results, and evaluate results with various metrics.

## Data Processing
[src/dataloaders/](src/dataloaders/) the following two data loaders:
- An **AAI Loader**, which is used to write and store AAI datasets, but also load them and prepare them for training our Slot Attention and AlignNet models. This was also extended to provide funtionality for loading the Spriteworld dataset as well.
- A **CLEVR-Hans Loader**, which is used to create the CLEVR-Hans3 dataset used, and load this when needed to train models.

## Model Implementations
[src/models/](src/models/) holds the implementations for the Slot Attention, Background Slot Attention, and SA+AlignNet models:

- **Slot Attention** builds object-centric representations of objects from an image inputs, and learns to decode these to be visualized.
- **Background Slot Attention** is an alternative form of Slot Attention which separates a background slot from the rest and sends this through a separate flow. In particular, it has its own simple deconvolutional decoder, which does not include spatial broadcasting.
- **AlignNet** learns to align Slot Attention slot representtaions through a dynamics and a permutation model.

## Plotting
[src/plots/](src/plots/) is comprised of plotting methods for both Slot Attention and AlignNet models. For Slot Attention, these are used to visualize input images, per-slot alpha masks, per-slot reconstructions, and full image reconstructions. For AlignNet, these are used to show aligned slots over time, or alignment plots showing slot-object assignmnets over time.


## Utils
[src/utils/](src/utils/) is composed of three main parts: The evalutaion metrics, the AAI ground-truth mask creator, and other common function and classes.

### Evaluation Metrics
There are three important evaluation metrics:
- The **Mean Squared Error (MSE)**, used to assess image reconstruction quality.
- The **Adjusted Rand Index (ARI)**, used to assess object representation quality.
- The **alignment score**, used to assess alignment accuracy.


### Ground-truth Mask Creator for AAI
The color-based mask creator is used to extract ground truth from AAI images, using HSV color ranges inherently present in the AAI arena configurations to differentiate between objects. These are needed to calculate ARI scores, and supervised object segmentation approaches are impossible without provided segmentations or labels. 


### Other Common Functions and Classes
There are also a few functions and classes that are repeatedly used throughout the codebase, so they are centralized here to avoid duplications.