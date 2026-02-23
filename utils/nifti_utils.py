"""NIfTI loading, saving, and resampling utilities."""

import numpy as np


def load_nifti(filepath):
    """Load a NIfTI file and return volume, affine, and header.

    Args:
        filepath: Path to .nii or .nii.gz file.

    Returns:
        volume: 3D numpy array.
        affine: 4x4 affine matrix.
        header: NIfTI header object.
    """
    import nibabel as nib

    img = nib.load(str(filepath))
    volume = img.get_fdata().astype(np.float32)
    return volume, img.affine, img.header


def save_nifti(volume, affine, filepath):
    """Save a 3D volume as NIfTI.

    Args:
        volume: 3D numpy array.
        affine: 4x4 affine matrix.
        filepath: Output path (.nii or .nii.gz).
    """
    import nibabel as nib

    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(img, str(filepath))


def resample_nifti(volume, old_affine, new_spacing, order=1):
    """Resample a NIfTI volume to a new isotropic spacing.

    Args:
        volume: 3D numpy array.
        old_affine: Original 4x4 affine.
        new_spacing: Target voxel spacing (scalar for isotropic).
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        resampled: Resampled 3D array.
        new_affine: Updated affine matrix.
    """
    from scipy.ndimage import zoom

    old_spacing = np.abs(np.diag(old_affine)[:3])
    zoom_factors = old_spacing / new_spacing

    resampled = zoom(volume, zoom_factors, order=order)

    new_affine = old_affine.copy()
    for i in range(3):
        new_affine[:3, i] = old_affine[:3, i] / zoom_factors[i] * np.sign(old_affine[i, i] + 1e-8)

    return resampled, new_affine


def voxel_to_world(affine, i, j, k):
    """Convert voxel indices to world coordinates."""
    voxel = np.array([i, j, k, 1.0])
    world = affine @ voxel
    return world[:3]


def world_to_voxel(affine, x, y, z):
    """Convert world coordinates to voxel indices."""
    inv_affine = np.linalg.inv(affine)
    world = np.array([x, y, z, 1.0])
    voxel = inv_affine @ world
    return voxel[:3]
