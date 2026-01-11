import os

import numpy as np
import matplotlib.pyplot as plt


def plot_belief_heatmap(hist, objid, grid_shape=None, save_path=None, show=False):
    """Plot a heatmap for a single object's histogram belief.

    Args:
        hist (dict): state -> prob; state must support state["pose"] returning (x,y) or (x,y,theta).
        objid: object id (used only for title/filename).
        grid_shape (tuple): optional (width, length). If None, inferred from poses.
        save_path (str): optional path to save PNG. If None and show=False, saves to "belief_obj{objid}.png".
        show (bool): if True, plt.show(); otherwise close after save.
    """
    if hist is None or len(hist) == 0:
        return

    # Infer grid size if not provided
    poses = [state["pose"] for state in hist]
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    width = max(xs) + 1 if grid_shape is None else grid_shape[0]
    length = max(ys) + 1 if grid_shape is None else grid_shape[1]

    heat = np.zeros((width, length))
    for state, prob in hist.items():
        px, py = state["pose"][:2]
        heat[px, py] = prob

    plt.figure(figsize=(6, 5))
    plt.imshow(heat.T, origin="lower", cmap="viridis")
    plt.colorbar(label="Belief")
    plt.title(f"Belief heatmap for obj {objid}")
    plt.xlabel("x")
    plt.ylabel("y")

    outfile = save_path if save_path is not None else f"belief_obj{objid}.png"
    plt.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()