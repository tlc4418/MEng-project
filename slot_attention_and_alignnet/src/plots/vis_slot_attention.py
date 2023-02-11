import pickle

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("seaborn-paper")
import io

import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf

from slot_attention_and_alignnet.src.utils import denorm


def normalize_across_slots(x):
    x_min, x_max = x.min((1, 2, 3), keepdims=True), x.max((1, 2, 3), keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    return x.clip(0, 1)


def plot_slot_attention(
    source,
    attn_masks,
    recos=None,
    masks=None,
    source_not_rescaled=True,
    recos_not_rescaled=True,
    show_time_steps=True,
    max_steps=7,
):
    """
    Expects shapes of form [batch_size, (slots), width, height, channels]

    Has three modes of operation:
        Attention only: recos, masks = None
        +Recos&Masks: Recos, masks = jnp.array
        +Dynamics: Recos, Masks = (reco:jnp.array, reco_dyn: jnp.array)...
    """

    batch_size, num_slots = attn_masks.shape[:2]

    attn_masks = normalize_across_slots(attn_masks)
    if masks is not None:
        # Will treat masks and recos as lists to simplify control flow
        if not isinstance(masks, list):
            masks = [masks]
            recos = [recos]

        for i in range(len(masks)):
            masks[i] = jnp.squeeze(masks[i])
            masks[i] = normalize_across_slots(masks[i])  # normalize along slot dims
            if recos_not_rescaled:
                recos[i] = denorm(recos[i])

    if source_not_rescaled:
        source = denorm(source)

    rows = num_slots + 1
    col_mult = 1 + len(masks) if masks is not None else 1
    cols = min(batch_size, max_steps)

    fig = plt.figure(
        figsize=(
            cols * col_mult**2,
            int(col_mult * 0.95 * rows) + (num_slots - 1 if col_mult == 1 else 0),
        )
    )
    gs_outer = gridspec.GridSpec(1, min(batch_size, max_steps), wspace=0.15, hspace=0.0)

    row1_titles = ["Source"] + (
        ["Reco"] if len(masks) < 2 else ["Reco Dyn", "Reco Perm"]
    )
    row2_titles = ["Attention"] + (
        ["Alpha Reco"] if len(masks) < 2 else ["Alpha Reco Dyn", "Alpha Reco Perm"]
    )
    for t in range(0, cols):
        gs_col = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            height_ratios=[1, num_slots],
            subplot_spec=gs_outer[0, t],
            wspace=0.0,
            hspace=0.10,
        )
        gs_col_row_1 = gs_col[0, 0].subgridspec(1, col_mult, wspace=0.01, hspace=0.0)

        for i in range(col_mult):
            ax = fig.add_subplot(gs_col_row_1[0, i])
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            ax.set_title(row1_titles[i])
            ax.imshow(source[t] if i == 0 else recos[i - 1][t])

            if show_time_steps:
                if col_mult == 3 and i == 1:
                    fig.text(
                        ax.get_position().x0,
                        0.92,
                        f"Step $t_{t}$",
                        ha="center",
                        c="k",
                        fontsize=14,
                    )
                elif (col_mult == 2 and i == 1) or (col_mult == 1):
                    x_start, x_end = ax.get_position().x0, ax.get_position().x1
                    fig.text(
                        x_start + (x_end - x_start) / 2,
                        0.91,
                        f"Step $t_{t}$",
                        ha="center",
                        c="k",
                        fontsize=14,
                    )

        gs_col_row_2 = gs_col[1, 0].subgridspec(
            num_slots, col_mult, wspace=0.01, hspace=0.0
        )
        for row in range(num_slots):
            for col in range(col_mult):
                ax = fig.add_subplot(gs_col_row_2[row, col])
                ax.tick_params(
                    bottom=False, labelbottom=False, left=False, labelleft=False
                )
                if row == 0:
                    ax.set_title(row2_titles[col])
                if t == 0 and col == 0:
                    ax.set_ylabel(f"Slot {row}", fontweight="bold")

                if col == 0:
                    ax.imshow(attn_masks[t, row], vmin=0, vmax=1)
                else:
                    ax.imshow(masks[col - 1][t, row], cmap="Blues_r", vmin=0, vmax=1)

    return fig, "Masks and stuff"


## Aditional plot functions for report ##

# Same plot but horizontal
def plot_slot_attention_horiz(
    source,
    attn_masks,
    recos=None,
    masks=None,
    slot_recos=None,
    source_not_rescaled=True,
    recos_not_rescaled=True,
    show_time_steps=True,
    max_steps=4,
):
    """
    Expects shapes of form [batch_size, (slots), width, height, channels]

    Has three modes of operation:
        Attention only: recos, masks = None
        +Recos&Masks: Recos, masks = jnp.array
        +Dynamics: Recos, Masks = (reco:jnp.array, reco_dyn: jnp.array)...
    """

    batch_size, num_slots = attn_masks.shape[:2]

    attn_masks = normalize_across_slots(attn_masks)
    if masks is not None:
        # Will treat masks and recos as lists to simplify control flow
        if not isinstance(masks, list):
            masks = [masks]
            recos = [recos]

        for i in range(len(masks)):
            masks[i] = jnp.squeeze(masks[i])
            masks[i] = normalize_across_slots(masks[i])  # normalize along slot dims
            if recos_not_rescaled:
                recos[i] = denorm(recos[i])

    if source_not_rescaled:
        source = denorm(source)

    cols = num_slots + 1
    row_mult = (
        1
        + (len(masks) if masks is not None else 1)
        + (1 if slot_recos is not None else 0)
    )
    rows = min(batch_size, max_steps)

    fig = plt.figure(
        figsize=(
            int(row_mult * 0.95 * cols) + (num_slots - 1 if row_mult == 1 else 0),
            rows * row_mult**2,
        )
    )
    gs_outer = gridspec.GridSpec(min(batch_size, max_steps), 1, wspace=0.0, hspace=0.15)

    col1_titles = ["Source"] + (
        ["Reco"] if len(masks) < 2 else ["Reco Dyn", "Reco Perm"]
    )
    col2_titles = (
        ["Attention"]
        + (["Alpha Mask"] if len(masks) < 2 else ["Alpha Reco Dyn", "Alpha Reco Perm"])
        + (["Slot Reco"] if slot_recos is not None else [])
    )
    for j, t in enumerate([0, 1, 2, 3]):
        gs_row = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            width_ratios=[1, num_slots],
            subplot_spec=gs_outer[j, 0],
            wspace=0.10,
            hspace=0.0,
        )
        gs_row_col_1 = gs_row[0, 0].subgridspec(
            row_mult - (1 if slot_recos is not None else 0), 1, wspace=0.0, hspace=0.01
        )

        for i in range(row_mult - (1 if slot_recos is not None else 0)):
            ax = fig.add_subplot(gs_row_col_1[i, 0])
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            ax.set_ylabel(col1_titles[i], fontsize=12)
            ax.imshow(source[t] if i == 0 else recos[i - 1][t])

            # Not checked
            if show_time_steps:
                if row_mult == 3 and i == 1:
                    fig.text(
                        ax.get_position().x0,
                        0.92,
                        f"Step $t_{t}$",
                        ha="center",
                        c="k",
                        fontsize=14,
                    )
                elif (row_mult == 2 and i == 1) or (row_mult == 1):
                    x_start, x_end = ax.get_position().x0, ax.get_position().x1
                    fig.text(
                        x_start + (x_end - x_start) / 2,
                        0.91,
                        f"Step $t_{t}$",
                        ha="center",
                        c="k",
                        fontsize=14,
                    )

        gs_row_col_2 = gs_row[0, 1].subgridspec(
            row_mult, num_slots, wspace=0.0, hspace=0.01
        )
        for col in range(num_slots):
            for row in range(row_mult):
                ax = fig.add_subplot(gs_row_col_2[row, col])
                ax.tick_params(
                    bottom=False, labelbottom=False, left=False, labelleft=False
                )
                if col == 0:
                    ax.set_ylabel(col2_titles[row], fontsize=12)
                if row == 0:
                    ax.set_title(f"Slot {col}", fontweight="bold", fontsize=10)

                if row == 0:
                    ax.imshow(attn_masks[t, col], vmin=0, vmax=1)
                else:
                    if slot_recos is not None and row == 2:
                        ax.imshow(denorm(slot_recos[t, col]))
                    else:
                        ax.imshow(
                            masks[row - 1][t, col], cmap="Blues_r", vmin=0, vmax=1
                        )
    return fig, "Masks and stuff"


# Plot only alpha masks horizontal
def plot_slot_attention_horiz_single(
    source,
    attn_masks,
    recos=None,
    masks=None,
    slot_recos=None,
    source_not_rescaled=True,
    recos_not_rescaled=True,
    show_time_steps=True,
    max_steps=4,
):
    """
    Expects shapes of form [batch_size, (slots), width, height, channels]

    Has three modes of operation:
        Attention only: recos, masks = None
        +Recos&Masks: Recos, masks = jnp.array
        +Dynamics: Recos, Masks = (reco:jnp.array, reco_dyn: jnp.array)...
    """

    batch_size, num_slots = attn_masks.shape[:2]

    attn_masks = normalize_across_slots(attn_masks)
    if masks is not None:
        # Will treat masks and recos as lists to simplify control flow
        if not isinstance(masks, list):
            masks = [masks]
            recos = [recos]

        for i in range(len(masks)):
            masks[i] = jnp.squeeze(masks[i])
            masks[i] = normalize_across_slots(masks[i])  # normalize along slot dims
            if recos_not_rescaled:
                recos[i] = denorm(recos[i])

    if source_not_rescaled:
        source = denorm(source)

    cols = num_slots + 2
    row_mult = 1
    rows = min(batch_size, max_steps)

    fig = plt.figure(
        figsize=(
            int(row_mult * 0.95 * cols) * 2.5,
            rows * row_mult**2 * 3,
        )
    )
    gs_outer = gridspec.GridSpec(min(batch_size, max_steps), 1, wspace=0.0, hspace=0.15)

    col1_titles = ["Source"] + ["Reco"]

    for j, t in enumerate([0, 1, 2, 3]):
        gs_row = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            width_ratios=[2.05, num_slots],
            subplot_spec=gs_outer[j, 0],
            wspace=0.05,
            hspace=0.0,
        )
        gs_row_col_1 = gs_row[0, 0].subgridspec(row_mult, 2, wspace=0.05, hspace=0.01)

        for i in range(row_mult + 1):
            ax = fig.add_subplot(gs_row_col_1[0, i])
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            ax.set_title(col1_titles[i], fontsize=12)
            ax.imshow(source[t] if i == 0 else recos[i - 1][t])

        gs_row_col_2 = gs_row[0, 1].subgridspec(
            row_mult, num_slots, wspace=0.0, hspace=0.01
        )
        for col in range(num_slots):
            for row in range(row_mult):
                ax = fig.add_subplot(gs_row_col_2[row, col])
                ax.tick_params(
                    bottom=False, labelbottom=False, left=False, labelleft=False
                )
                if row == 0:
                    ax.set_title(f"Slot {col}", fontweight="bold", fontsize=10)
                    ax.imshow(masks[row - 1][t, col], cmap="Blues_r", vmin=0, vmax=1)

    print("DONE")
    return fig, "Masks and stuff"


# Plot aligned slot reconstructions, with horizontal time dimension
def plot_slot_attention_horiz_single_align(
    source,
    attn_masks,
    recos=None,
    masks=None,
    slot_recos=None,
    source_not_rescaled=True,
    recos_not_rescaled=True,
    show_time_steps=True,
    max_steps=4,
):
    """
    Expects shapes of form [batch_size, (slots), width, height, channels]

    Has three modes of operation:
        Attention only: recos, masks = None
        +Recos&Masks: Recos, masks = jnp.array
        +Dynamics: Recos, Masks = (reco:jnp.array, reco_dyn: jnp.array)...
    """

    batch_size, num_slots = attn_masks.shape[:2]

    attn_masks = normalize_across_slots(attn_masks)
    if masks is not None:
        # Will treat masks and recos as lists to simplify control flow
        if not isinstance(masks, list):
            masks = [masks]
            recos = [recos]

        for i in range(len(masks)):
            masks[i] = jnp.squeeze(masks[i])
            masks[i] = normalize_across_slots(masks[i])  # normalize along slot dims
            if recos_not_rescaled:
                recos[i] = denorm(recos[i])

    if source_not_rescaled:
        source = denorm(source)

    cols = num_slots + 2
    row_mult = 1
    rows = min(batch_size, max_steps)

    fig = plt.figure(
        figsize=(
            int(row_mult * 0.95 * cols) * 2.5,
            rows * row_mult**2 * 2.2,
        )
    )
    gs_outer = gridspec.GridSpec(min(batch_size, max_steps), 1, wspace=0.0, hspace=0.05)

    col1_titles = ["Source"] + ["Reco"]

    for j, t in enumerate([0, 4, 5, 6]):
        gs_row = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            width_ratios=[2.05, num_slots],
            subplot_spec=gs_outer[j, 0],
            wspace=0.03,
            hspace=0.0,
        )
        gs_row_col_1 = gs_row[0, 0].subgridspec(row_mult, 2, wspace=0.06, hspace=0.01)

        for i in range(row_mult + 1):
            ax = fig.add_subplot(gs_row_col_1[0, i])
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            if j == 0:
                ax.set_title(col1_titles[i], fontsize=14)
            if i == 0:
                ax.set_ylabel(f"t={t}", fontsize=15)
            ax.imshow(source[t] if i == 0 else recos[i][t])

        gs_row_col_2 = gs_row[0, 1].subgridspec(
            row_mult, num_slots, wspace=0.0, hspace=0.01
        )
        for col in range(num_slots):
            for row in range(row_mult):
                ax = fig.add_subplot(gs_row_col_2[row, col])
                ax.tick_params(
                    bottom=False, labelbottom=False, left=False, labelleft=False
                )

                if row == 0:
                    if j == 0:
                        ax.set_title(f"Slot {col}", fontweight="bold", fontsize=12)

                    ax.imshow(
                        denorm(slot_recos[t, col])
                        # masks[row - 1][t, col], cmap="Blues_r", vmin=0, vmax=1
                    )
    return fig, "Masks and stuff"
