"""3D medical image visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt


def plot_orthogonal_slices(volume, slices=None, title='', cmap='gray',
                           figsize=(15, 5), vmin=None, vmax=None):
    """Plot axial, coronal, and sagittal slices through a 3D volume.

    Args:
        volume: 3D numpy array (Z, Y, X) or (slices, rows, cols).
        slices: Tuple of (z, y, x) slice indices. Defaults to center.
        title: Figure title.
        cmap: Colormap name.
        figsize: Figure size.
        vmin, vmax: Display range.
    """
    vol = np.asarray(volume)
    if slices is None:
        slices = tuple(s // 2 for s in vol.shape)
    z, y, x = slices

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(vol[z, :, :], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Axial (z={z})')
    axes[0].axhline(y=y, color='r', linewidth=0.5, alpha=0.7)
    axes[0].axvline(x=x, color='g', linewidth=0.5, alpha=0.7)

    axes[1].imshow(vol[:, y, :], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'Coronal (y={y})')
    axes[1].axhline(y=z, color='r', linewidth=0.5, alpha=0.7)
    axes[1].axvline(x=x, color='g', linewidth=0.5, alpha=0.7)

    axes[2].imshow(vol[:, :, x], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title(f'Sagittal (x={x})')
    axes[2].axhline(y=z, color='r', linewidth=0.5, alpha=0.7)
    axes[2].axvline(x=y, color='g', linewidth=0.5, alpha=0.7)

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_interactive_slices(volume, axis=0, cmap='gray', figsize=(8, 8)):
    """Create an interactive slice viewer with ipywidgets slider.

    Args:
        volume: 3D numpy array.
        axis: Axis to slice along (0=axial, 1=coronal, 2=sagittal).
        cmap: Colormap.
        figsize: Figure size.
    """
    from ipywidgets import interact, IntSlider

    axis_names = {0: 'Axial', 1: 'Coronal', 2: 'Sagittal'}
    n_slices = volume.shape[axis]
    vmin, vmax = volume.min(), volume.max()

    def show_slice(idx):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        slc = np.take(volume, idx, axis=axis)
        ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'{axis_names[axis]} slice {idx}/{n_slices - 1}')
        ax.axis('off')
        plt.show()

    interact(show_slice, idx=IntSlider(
        min=0, max=n_slices - 1, step=1,
        value=n_slices // 2,
        description=f'{axis_names[axis]}:'
    ))


def plot_comparison_grid(images, titles, rows=1, cmap='gray',
                         figsize=None, vmin=None, vmax=None):
    """Plot a grid of 2D images for comparison.

    Args:
        images: List of 2D arrays.
        titles: List of titles.
        rows: Number of rows.
        cmap: Colormap.
        figsize: Figure size (auto-calculated if None).
    """
    n = len(images)
    cols = (n + rows - 1) // rows
    if figsize is None:
        figsize = (5 * cols, 5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        axes[r, c].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[r, c].set_title(title)
        axes[r, c].axis('off')

    # Hide empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    return fig
