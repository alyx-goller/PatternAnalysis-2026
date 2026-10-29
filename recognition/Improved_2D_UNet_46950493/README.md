# 2D Improved U-Net Research - Segmentation Anatomical Contouring and Radiotherapy Planning

**Name:** Alyx Goller
**Student ID:** 4695049

**Open-Ended Project and Difficulty**
2.2 Segmentation: Anatomical Contouring and Radiotherapy Planning
2D Improved U-Net (Normal Difficulty)


## Introduction

### Algorithm and the problem it solves
Radiotherapy planning needs the prostate outlined accurately on MRI, but a single
global Dice score can hide local failures: a model may score well on the wide central
slices yet fail at the base and apex, or leak across the thin boundary into the rectum
or bladder, where errors have direct clinical cost. This project segments HipMRI 2D
slices into six classes (labels 0-5, with class 5 the prostate) and asks whether an
improved U-Net gives better segmentation than the original 2D U-Net (Ronneberger et
al., 2015), and at what computing cost. The prostate covers only a tiny fraction of
each slice, so the model must cope with severe class imbalance. The target is a Dice
score of at least 0.75 on the prostate label on the held-out test set.

### How it works

The prototype model created for this task is a 2D adaptation of the improved U-Net of Isensee et al. (2017). The encoder-decoder (U-shaped) structure follows:
**Encoder:**
Five levels with 16, 32, 64, 128 and 256 filters/channels created at each output level. This process is structured as:
- Initial channel configuration - 3x3 Convolution
4 x layers of:
- Context Block: input -> (norm, act, conv) -> dropout -> (norm, act, conv) -> add output to input (residual connection)
- Downsampling Block: 3x3 stride 2 Convolution
**Bottleneck**
The end of the encoder block is the bottleneck which includes a fifth and final context block (256 channels, 16×8) holding the most abstract, widest-context features.
**Decoder**
4 x layers of:
- Upsampling Block: conv -> norm -> act (double feature map size)
- Concatenate output with encoder output at the same level (skip connection)
- Localization Block: 3x3conv -> norm -> act -> 1x1 conv -> norm -> act:
    3x3 ConvNormAct (Combine nearby features from skip and feature map combination)
    1x1 ConvNormAct (reduce double channels from concat back to decoder process order)
**Final Segmentation**
:is an interpolation of the final three localization block outputs (deep supervision) to give the final per-pixel class scores, which a softmax turns into probabilities applied in the loss; argmax at prediction

**Loss Function**
Training uses a multi-class Dice loss, which weights the small prostate class equally with large background classes, with the Adam optimiser (learning rate 5e-4, decayed by 0.985 per epoch).
Each slice is z-score normalised, labels are one-hot encoded, and batches are padded to a multiple of 16 so the four downsampling steps divide evenly.

conv = 3x3 convolution stride 1 (unless stated otherwise)
norm = InstanceNorm
act = leaky ReLu activation function

This process is an adaptation of the improved U-Net of Isensee et al. (2017). which follows the following structure:
![Improved 2D U-Net architecture](figures/Isensee_unet_structure)


### File Breakdown

**Dataset.py:**
Loads the NIfTI slices, pairs images with labels, z-score normalises, one-hot encodes the masks, pads batches to a multiple of 16 for cases where images are of different sizes, builds the train/validation/test DataLoaders.
**Modules.py:**
Network blocks (ConvNormAct, context, upsampling, localization), the improved `UNet2D`, `DiceLoss` and `dice_counts` for performance evaluation and training loss.
**Train.py:**
Training and validation loop, learning-rate schedule, saves the best checkpoint by validation prostate Dice.
**Predict.py:**
Test-set evaluation (per-class Dice, over/under-contoured %), best/worst-batch visualisation and the class-imbalance slice test.
**Figures folder:**
Figures used in README including model architecture and visualization examples.



## Feasibility Report

### User Need, Scope & Acceptance Criteria:
**The intended user:** 
The complete model would be used by an external-beam radiotherapy professional requiring millimetre-level delineation of the prostate clinical target volume while tightly sparing adjacent critical Organs at Risk.
**Prototype purpose:** 
The prototype is built to compare performance and computational cost to an older segmentation model. This is to determine whether increased accuracy is worth the cost of updating the model.
**Four testable acceptance criteria addressing the domain’s open dilemma:**
1.	Comparing loss improvement over increasing epochs
2.	Comparing run time of training between newer and older models
3.	Visual analysis of input image, ground truth mask, predicted mask – discover if improved model can reduce over-contouring and under-contouring more than older model 
4.	Establishing whether prototype meets 0.75 target Dice Score (or more)

### Model Choice & Course Concepts:
**The proposed dataset configuration:** 
Keras slices split into data and segmentation label files ensuring patient level splitting and no leakage between training, validation or testing data.
**Chosen difficulty band/model architecture:**
Improved 2D U-Net – normal difficulty
**Course concepts and literature impact:**
Literature between ref [1] and ref [2] from COMP3710 report task encouraged comparison between the 2D U-Net and an improved version while modifying the improved version from 3D OASIS to HipMRI data.
Course content created interest in segmentation through lectures and provided supporting CNN code for the model’s basic structure.
**Preliminary Feasibility Evidence:**
The following is the output of an initial training test using a subset of the data. The purpose of this preliminary test is to confirm runtime speed with the provided resources, a decreasing loss value at each epoch and to ensure the Dice performance can meet the established target.

Quick test: 2000 training slices / 400 validation slices - device: NVIDIA A100-PCIE-40GB
Epoch   1 (13s) | train loss 0.3901 | val loss 0.2714 | prostate Dice 0.6988 | all classes [0.953, 0.921, 0.726, 0.584, 0.569, 0.699]

Epoch   2 (11s) | train loss 0.1828 | val loss 0.1810 | prostate Dice 0.7044 | all classes [0.982, 0.947, 0.801, 0.714, 0.807, 0.704]

Epoch   3 (11s) | train loss 0.1280 | val loss 0.1516 | prostate Dice 0.7585 | all classes [0.976, 0.961, 0.853, 0.755, 0.812, 0.758]

Epoch   4 (11s) | train loss 0.1038 | val loss 0.1324 | prostate Dice 0.7791 | all classes [0.985, 0.967, 0.882, 0.768, 0.845, 0.779]

Epoch timing reduced once files completely loaded – overall, the 50-epoch run took approximately 6 minutes.

### Risks, Budget & Fallback:
**Key technical risks:**
Inconsistent input sizes – 
During testing, images with different widths were found and padding was added to adjust to the size difference for even inputs and outputs through the model. The padding should not have much of an impact, but it may affect InstanceNorm2d inside the hidden layers since the images mean and standard deviation is taken without excluding any padding.

Small target class/Class imbalance –
The prostate is a small portion of the image slices, and some slices may exclude this class (or others) altogether. Dice loss and a smoothing term (epsilon) is used but the loss figure printed is still an average over many images (including slices which exclude the prostate).

**Computing Budget:**
Hardware -
One type of GPU (NVIDIA A100) is available for testing, and it may have a queue to use and time limits on usage. The paper used for comparison to the basic 2D U-net model uses a NVidia Titan GPU with 6 GB of memory, which makes the computational cost not perfectly comparable. Ref: [1].

Dataset – The available dataset of 2D HipMRI images and labels includes:
| Train Set | Validate Set | Test Set |
| :---: | :---: | :---: |
| 11460 images | 660 images | 540 images |

The comparison study uses 30 training images and relies on strong use of data augmentation. Ref: [1]. 

Timeline –
A restricted amount of time is dedicated to this task due to university timelines. Training, improvement capacity, task feedback and comparison detail may be limited. The improved U-Net changes several things at once (instance norm, residual blocks, strided convolution down sampling, deep supervision, leaky ReLU activation, up sampling with repeat pixels then 3x3 convolution, localization in decoder, filter width 16-256). Determining the performance and cost of each of these changes individually to determine the best balance is not possible in the available timeframe.

**Planned Experiment:**
Experimental process to confirm feasibility and progress of prototype

1. Baseline test - After programming Improved 2D U-net model, run training and validation to confirm loss function is reducing over multiple epochs and that DICE score is approaching or achieving 0.75 Dice score goal
2. Subset Analysis - Run training on a subset of data over a larger epoch range to discover where training is plateuing and if any setting changes can improve learning efficiency.
3.	Full dataset and weight storing - Discover training progress on entire training dataset and large number of epochs. Determine if larger dataset changes results in performance and what computational cost of max Dice score is. Store best weights for predictions on test dataset.
5.	Prediction and Visualization: With best performing model weights, run prediction on unseen test dataset, visualize performance, compare findings to older 2D U-net model and visualize fail cases and any potential limitations.

**Workable Fallback:**

**If the improved model doesn’t reach 0.75 on the test set:** 
Report the best result honestly and analyse why this may be the case by comparing per-class Dice scores, the training and validation loss score gap and identifying failure cases.

**If timeline runs short or workload underestimated:**
Completion of test program, full dataset training with limited epochs and a final test dice figure will be prioritized for the most accurate comparison.

**If improved architecture is unstable or too expensive (at assignment completion):**
Future analysis of the individual differences between the improved 2D-U-Net and original 2D U-net may help find the best balance of performance and cost. Further study into 2D CAN models may present preferable findings.


## Findings and Prototype Evaluation

**Baseline and Subset Test:**

The loss function did fall considerably within smaller dataset tests. Dice scores increased by approxiamtely ___ per epoch, reaching 0.__ by epoch __. With a subset of __ training images and __ validation images, a peak Dice score of __ was achieved, already exceding the target Dice value.
The validation loss scores plateued around epoch __ and stayed within __ - __ range. At this stage, a larger dataset was considered as a potential solution of reducing the validation loss further and would be evaluated in the next training test.

The selected initial learning rate, learning rate decay and weigth decay values were ___, ___ and __, respectively.


**Full Dataset Test:**

The full dataset, as stated in the feasibility report, includes 11460 training images and 660 validation images. 
Initially, 11 epochs were run, taking a total time of approximately 18 minutes. A second instance of training was tested to see if more epochs would improve the validation prostate dice score and in turn the best weights. 
The second training instance took approximately 21 minutes and 23 seconds. In this time 19 epochs completed and the lowest validation score was 0.1029, occuring at epoch 14 and the highest prostate Dice score was 0.8404, occuring at epoch 9.

2. 
Sufficiently documenting your project, working principles, feasibility review, experiments,
and instructions 