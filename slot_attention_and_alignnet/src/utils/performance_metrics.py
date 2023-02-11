import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree
import cv2
import imutils
import math
import matplotlib.pyplot as plt
import tensorflow as tf


def mse(source, reconstruction):
    return jnp.mean((source - reconstruction) ** 2)


def ari_score(true_masks, pred_masks, max_process=100):
    # Filter out samples with only 1 cluster
    valid_indices = jnp.asarray(
        np.fromiter(map(mask_num_check, true_masks), dtype=bool)
    )
    pred_masks = pred_masks[valid_indices]
    true_masks = true_masks[valid_indices]

    # Preprocess batch shape
    batch_size = true_masks.shape[0]
    num_processed = min(batch_size, max_process)
    pred_masks = preprocess_batch(pred_masks, batch_size)[:num_processed]
    true_masks = preprocess_batch(true_masks, batch_size)[:num_processed]

    ari = adjusted_rand_index(true_masks, pred_masks)
    return jnp.mean(ari), num_processed


def adjusted_rand_index(true_mask, pred_mask):
    r"""Computes the adjusted Rand index (ARI), a clustering similarity score.
    This implementation ignores points with no cluster label in `true_mask` (i.e.
    those points for which `true_mask` is a zero vector). In the context of
    segmentation, that means this function can ignore points in an image
    corresponding to the background (i.e. not to an object).
    Args:
    true_mask: JNP array of shape [batch_size, n_points, n_true_groups].
        The true cluster assignment encoded as one-hot.
    pred_mask: JNP array of shape [batch_size, n_points, n_pred_groups].
        The predicted cluster assignment encoded as categorical probabilities.
        This function works on the argmax over axis 2.
    Returns:
    ARI scores as a JNP array of shape [batch_size].
    Raises:
    ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
        We've chosen not to handle the special cases that can occur when you have
        one cluster per datapoint (which would be unusual).
    References:
    Adapted from Multi-Object Datasets
        https://github.com/deepmind/multi_object_datasets
        # ============================================================================
        # Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #    http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.
        # ============================================================================
    """

    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    if n_points <= n_true_groups and n_points <= n_pred_groups:
        # This rules out the n_true_groups == n_pred_groups == n_points
        # corner case, and also n_true_groups == n_pred_groups == 0, since
        # that would imply n_points == 0 too.
        # The sklearn implementation has a corner-case branch which does
        # handle this. We chose not to support these cases to avoid counting
        # distinct clusters just to check if we have one cluster per datapoint.
        raise ValueError(
            "adjusted_rand_index requires n_groups < n_points. We don't handle "
            "the special cases that can occur when you have one cluster "
            "per datapoint."
        )

    pred_group_ids = jnp.argmax(pred_mask, -1)
    # We convert true and predicted clusters to one-hot ('oh') representations.
    true_mask_oh = jnp.array(true_mask, jnp.float32)  # already one-hot
    pred_mask_oh = jax.nn.one_hot(pred_group_ids, n_pred_groups)  # returns float32

    n_points = true_mask_oh.sum((1, 2))

    nij = jnp.einsum("bji,bjk->bki", pred_mask_oh, true_mask_oh)
    a = jnp.sum(nij, axis=1)
    b = jnp.sum(nij, axis=2)

    rindex = jnp.sum(nij * (nij - 1), axis=[1, 2])
    aindex = jnp.sum(a * (a - 1), axis=1)
    bindex = jnp.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2
    denom = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denom

    # The case where n_true_groups == n_pred_groups == 1 needs to be
    # special-cased (to return 1) as the above formula gives a divide-by-zero.
    return jax.ops.index_update(ari, denom == 0, 1)


def mask_num_check(masks, threshold=1):
    return sum(map(lambda x: jnp.count_nonzero(x) > 1, masks)) > threshold


def preprocess_batch(batch, batch_size):
    batched = batch.squeeze().reshape(batch_size, -1, 128, 128)
    return batched.transpose(0, 2, 3, 1).reshape(batch_size, 128 * 128, -1)


def alignment_score(attn):
    # Slot object assignment has shape [B, T, Slots]
    slot_object_assignment = get_slot_assignments(attn)

    # Calculate alignment by comparing slot object assignments at each timestep with the previous timestep
    ret = (
        jnp.count_nonzero(
            (slot_object_assignment[:, :-1] - slot_object_assignment[:, 1:]) == 0
        )
        / slot_object_assignment[:, 1:].size
    )
    return ret


def get_nearest_neighbors(ground_truth_posns, slotattn_posns):
    ground_truth_posns = np.array(ground_truth_posns, np.float32)

    # Construct KDTrees from positions and query nearest neighbours
    tree = KDTree(ground_truth_posns)
    _, slot_ids = tree.query(slotattn_posns)  # <-- the nearest point
    return slot_ids


def get_object_positions(attn: jnp.array, resolution, num_slots, time_steps):
    attn = attn.reshape(-1, num_slots, *resolution)
    l = np.array(list(map(_get_object_position, attn)))
    l = jnp.reshape(l, (-1, time_steps, num_slots, 2))
    return l


def _get_object_position(attn_mask: jnp.array, res: int = 128) -> jnp.array:
    """Use OpenCV to get contours from attention masks (single time step, not batched)
        and then use image moments to find the centers of these contours.
        Based on https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
            and https://stackoverflow.com/questions/57954206/how-to-obtain-a-dynamic-threshold-for-contour-detection-in-opencv
        The coordinate system goes 0->res, with (0,0) at top left, and (res-1,res-1) at bottom right

    Args:
        attn_mask (jnp.array): [description]
        res (int, optional): [description]. Defaults to RESOLUTION.

    Returns:
        jnp.array: [description]
    """
    num_slots = attn_mask.shape[0]
    attn_mask = attn_mask.reshape((num_slots, res, res)).copy()
    attn_mask = (attn_mask / attn_mask.max(axis=(1, 2))[:, None, None])[:, :, :, None]
    attn_mask = (attn_mask * 255.0).astype("uint8")

    slot_obj_posns = np.full([num_slots, 2], -1)

    MAX_CONTOUR_AREA = (res**2) / 3
    MIN_CONTOUR_AREA = (res**2) / 250
    for i in range(num_slots):
        blur = attn_mask[i]
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Filter out contours which are empty or just the whole image
        cnts = list(
            filter(
                lambda c: MIN_CONTOUR_AREA < cv2.contourArea(c) < MAX_CONTOUR_AREA, cnts
            )
        )

        # Slots containing less than 1 contour are discarded
        if len(cnts) < 1:
            continue
        elif len(cnts) > 1:
            cnts = [cnts[np.argmax([cv2.contourArea(c) for c in cnts])]]

        # Get center of contour
        M = cv2.moments(cnts[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        slot_obj_posns[i] = [cX, cY]

    return slot_obj_posns


def get_slot_assignments(attn: jnp.array):
    batch_size, time_steps, num_slots, res_x, res_y = attn.shape[:-1]

    # Get main object contour for all slots
    slot_obj_posns = get_object_positions(attn, [res_x, res_y], num_slots, time_steps)

    # Create assignments from slot to object id
    slot_assignments = np.full(shape=(batch_size, time_steps, num_slots), fill_value=-1)
    for b in range(batch_size):
        # At timestep 0, assign each slot containing an object a unique id
        object_id = 0
        for slot in range(num_slots):
            if slot_obj_posns[b, 0, slot, 0] != -1:
                slot_assignments[b, 0, slot] = object_id
                object_id += 1

        # For all further timesteps, assign the object id of the nearest neighbour from the previous timestep
        for time_step in range(1, time_steps):
            contours = slot_obj_posns[b, time_step]
            nearest_prev_slots = get_nearest_neighbors(
                slot_obj_posns[b, time_step - 1], contours
            )
            slot_assignments[b, time_step] = slot_assignments[b, time_step - 1][
                nearest_prev_slots
            ]

    return slot_assignments
