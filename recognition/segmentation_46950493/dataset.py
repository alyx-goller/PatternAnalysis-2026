from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

root = Path("/Users/alyxgoller/Desktop/COMP3710/HipMRI_data")   

def get_paths(folder):
    return sorted(Path(folder).glob("*.nii*"))

train_imgs = get_paths(root / "train_imgs")      # folder names: change to yours
train_labs = get_paths(root / "train_seg_labels")
val_imgs   = get_paths(root / "validate_imgs")
val_labs   = get_paths(root / "validate_seg_labels")
test_imgs  = get_paths(root / "test_imgs")
test_labs  = get_paths(root / "test_seg_labels")

# Sanity check: images and labels must pair up
for imgs, labs in [(train_imgs, train_labs), (val_imgs, val_labs), (test_imgs, test_labs)]:
    assert len(imgs) == len(labs), (len(imgs), len(labs))
    for i, l in list(zip(imgs, labs))[:3]:
        print(i.name, "<->", l.name)

#working : 21/10


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")



def load_slice(path):
    """Load one NIfTI file as a 2D float32 array (H, W).
       Return: The single image as an array
    """
    arr = nib.load(str(path)).get_fdata(caching="unchanged")
    #if 3D only keep first slice
    if arr.ndim == 3:
        arr = arr[:, :, 0]          # keep first slice if file is 3D
    return arr.astype(np.float32)



#turns label images into their binary masks -
#returned array is channel x (array/image shape) - one mask per channel showing
#the 1's portion of where the channel(channel = identified feature like prostate) is present in the image (0's everywhere else)
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
