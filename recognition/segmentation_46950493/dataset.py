import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

import numpy as np
import nibabel as nib
from tqdm import tqdm

#turns label images into their binary masks -
#returned array is channel x (array/image shape) - one mask per channel showing
#the 1's ratio of where the channel is present in the image (0's everywhere else)
#(channel = each of distinct object in image (label has already segmented))
"""
For example, a mask with background label `0` and prostate label `1`
becomes two channels: one marking background pixels, one marking prostate pixels.
"""
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    labels = np.unique(arr)
    result = np.zeros(arr.shape + (len(labels),), dtype=dtype)

    for channel, label in enumerate(labels):
        result[..., channel] = (arr == label)

    return result

#reads every file path in list given
"""
With `normImage=True`, it z-score normalizes each image.
 With `categorical=True`,it converts the input into label channels.
 With `getAffines=True`, it also returns spatial affine matrices.
 With `early_stop=True`, it stops early, leaving the unused preallocated entries as zeros
"""
def load_data_2D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    early_stop=False,
):
    """
    Load medical images from the provided filenames into a batch.

    If a NIfTI file is 3D, only its first slice is used. The output is
    pre-allocated for Conv2D input.

    normImage: z-score normalize each image.
    early_stop: stop loading after the first 22 images.
    """
    #loading image from voxel to real world space coordinates
    affines = []
    #number of images
    num = len(imageNames)


    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")
    if first_case.ndim == 3:
        first_case = first_case[:, :, 0]

    #if this is conversion to labels, get one hot encoded mask (to_channels)
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        nifti_image = nib.load(inName)
        inImage = nifti_image.get_fdata(caching="unchanged")
        affine = nifti_image.affine

        if inImage.ndim == 3:
            inImage = inImage[:, :, 0]

        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines



    return images
