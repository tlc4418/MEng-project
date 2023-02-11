import matplotlib.pyplot as plt

plt.style.use("seaborn-paper")
import colorsys
import numpy as np


def hsv_to_rgb(c):
    """Convert HSV tuple to RGB tuple."""
    return np.array(colorsys.hsv_to_rgb(*c))


def plot_alignment(slot_object_assignments, object_info):
    T, num_slots = slot_object_assignments.shape[:2]
    # Hardcoded 7 for report layout
    T = min(T, 7)
    f, ax = plt.subplots(1, 1, figsize=(T, num_slots))
    for i in range(num_slots - 1):
        ax.hlines(
            i + 0.5, -0.5, T - 0.5, color="gray", linestyle="dashed", linewidth=0.5
        )
    ax.set_ylim(-0.5, num_slots - 0.5)
    ax.set_yticks(range(num_slots))
    ax.set_ylabel("Slot", fontsize=20)
    ax.set_yticklabels(range(num_slots), fontsize=14)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xlabel("Time Step", fontsize=20)
    ax.set_xticks(range(T))
    ax.set_xticklabels(range(T), fontsize=14)
    ax.invert_yaxis()
    colors, shapes = object_info
    colors = colors / 255.0
    edges = list(
        map(hsv_to_rgb, list(np.minimum(np.maximum(colors, [0, 0.2, 0]), [1, 1, 0.8])))
    )
    colors = list(map(hsv_to_rgb, list(colors)))
    for t in range(T):
        for j in range(num_slots):  #
            if slot_object_assignments[t, j] != -1:
                # Slot-object Markers
                marker = shapes[slot_object_assignments[t, j]]
                color = colors[slot_object_assignments[t, j]]
                edgecolor = edges[slot_object_assignments[t, j]]
                ax.scatter(
                    t,
                    j,
                    marker=marker,
                    s=300,
                    color=color,
                    edgecolor=edgecolor,
                    linewidth=2,
                    alpha=1,
                    zorder=10,
                )

                # Next time step object matching
                if t < T - 1:
                    for t_start_match in range(t + 1, T):
                        slots_for_obj_next_step = np.argwhere(
                            slot_object_assignments[t_start_match, :]
                            == slot_object_assignments[t, j]
                        )

                        linestyle = "-" if t_start_match == t + 1 else "dashed"
                        for slot in slots_for_obj_next_step:
                            ax.plot(
                                (t, t_start_match),
                                (j, slot),
                                color=color,
                                linestyle=linestyle,
                                linewidth=2,
                                alpha=1,
                                zorder=1,
                            )

                        if len(slots_for_obj_next_step) > 0:
                            break
    plt.subplots_adjust(wspace=0, hspace=0)
    return f, "Alignment of an episode"
