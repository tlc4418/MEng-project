import cv2
import numpy as np
import scipy.ndimage
import skimage.feature as feature
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

# ---------- HSV color ranges per object type ---------- #
# Ramp
LIGHT_PINK = (130, 70, 145)
DARK_PINK = (175, 255, 255)

# Good goal
LIGHT_GREEN = (35, 20, 160)
DARK_GREEN = (60, 115, 235)

# Good goal multi
LIGHT_YELLOW = (23, 70, 180)
DARK_YELLOW = (30, 150, 255)

# Wall & cylinder tunnel & cardboard box
LIGHT_GRAY_1 = (85, 0, 0)
DARK_GRAY_1 = (115, 70, 164)
LIGHT_GRAY_2 = (14, 0, 80)
DARK_GRAY_2 = (55, 45, 165)
LIGHT_GRAY_3 = (70, 0, 165)
DARK_GRAY_3 = (110, 20, 215)


# ---------- Mask generators per object type ---------- #


def get_ramp_mask(hsv_img):
    return cv2.inRange(hsv_img, LIGHT_PINK, DARK_PINK)


def get_goodgoal_mask(hsv_img):
    return cv2.inRange(hsv_img, LIGHT_GREEN, DARK_GREEN)


def get_multigoal_mask(hsv_img):
    return cv2.inRange(hsv_img, LIGHT_YELLOW, DARK_YELLOW)


def get_grayobstacle_mask(hsv_img):
    # Several masks with denoising
    mask1 = cv2.inRange(hsv_img, LIGHT_GRAY_1, DARK_GRAY_1)
    mask1 = torch.tensor(np.array([mask1])).float() / 255
    mask1 = (
        np.array(F.avg_pool2d(mask1, kernel_size=3, stride=1, padding=1)[0]) > 0.4
    ).astype(np.uint8) * 255

    mask2 = cv2.inRange(hsv_img, LIGHT_GRAY_2, DARK_GRAY_2)
    mask2 = torch.tensor(np.array([mask2])).float() / 255
    mask2 = (
        np.array(F.avg_pool2d(mask2, kernel_size=3, stride=1, padding=1)[0]) > 0.4
    ).astype(np.uint8) * 255

    mask3 = cv2.inRange(hsv_img, LIGHT_GRAY_3, DARK_GRAY_3)

    # Combine masks due to the sesnsitive nature of these color ranges (very simialar to some bg elements)
    full_mask = mask1 + mask2 + mask3

    # Final avg pooling for denoising
    tensor_norm_mask = torch.tensor(np.array([full_mask])).float() / 255
    denoised_mask = (
        np.array(F.avg_pool2d(tensor_norm_mask, kernel_size=3, stride=1, padding=1)[0])
        > 0.4
    ).astype(np.uint8) * 255

    return denoised_mask


# ----------------------------------------------------- #

MASK_GENERATORS = [
    get_ramp_mask,
    get_goodgoal_mask,
    get_multigoal_mask,
    get_grayobstacle_mask,
]


def get_object_masks_from_segmentations(input_segmentation, size_threshhold=False):
    """Use FloodFill algorithm to retrieve all object masks from a color range segmentation"""
    object_masks = []
    inp = np.copy(input_segmentation)
    floodflags = 8  # num of nearest neighbours to be considered
    floodflags |= 1 << 8  # fill mask with this value

    M, N = inp.shape
    for i in range(M):
        for j in range(N):
            if inp[i, j] == 255:
                mask = np.zeros((M + 2, N + 2), dtype=np.uint8)
                cv2.floodFill(
                    image=inp, newVal=0, mask=mask, seedPoint=(j, i), flags=floodflags
                )
                mask = mask[1:-1, 1:-1]

                # Remove tiny objects, due to color range ambiguity
                if not size_threshhold or np.count_nonzero(mask) > 100:
                    object_masks.append(mask)

    return object_masks


def get_ground_truth_masks(img, max_masks):
    """Create object masks from an AAI image, using color-based segmentation and FloodFill"""
    M, N = img.shape[:2]
    object_masks = []
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    input_segmentations = [gen(hsv_img) for gen in MASK_GENERATORS]

    for i, seg in enumerate(input_segmentations):
        object_masks.extend(
            get_object_masks_from_segmentations(seg, i == 3)
        )  # i check for gray objects

    # Pad with empty masks to fill max_masks
    num_masks = len(object_masks)

    if num_masks > max_masks:
        object_masks.sort(key=np.count_nonzero, reverse=True)
        for i in range(num_masks - max_masks):
            object_masks.pop()

    for i in range(max_masks - num_masks):
        object_masks.append(np.zeros((M, N)))

    return object_masks
