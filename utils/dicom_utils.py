"""DICOM loading and conversion utilities."""

import numpy as np
from pathlib import Path


def load_dicom_series(directory, sort_by='SliceLocation'):
    """Load a DICOM series from a directory.

    Args:
        directory: Path to directory containing DICOM files.
        sort_by: DICOM tag to sort slices by (default: SliceLocation).

    Returns:
        volume: 3D numpy array (slices, height, width).
        slices: List of sorted pydicom Dataset objects.
    """
    import pydicom

    dcm_dir = Path(directory)
    dcm_files = list(dcm_dir.glob('*.dcm')) + list(dcm_dir.glob('*.DCM'))
    if not dcm_files:
        # Try loading all files (DICOM files sometimes lack extensions)
        dcm_files = [f for f in dcm_dir.iterdir() if f.is_file()]

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"No valid DICOM files found in {directory}")

    slices.sort(key=lambda s: float(getattr(s, sort_by, 0)))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    # Apply rescale slope/intercept if present
    if hasattr(slices[0], 'RescaleSlope'):
        slope = float(slices[0].RescaleSlope)
        intercept = float(slices[0].RescaleIntercept)
        volume = volume * slope + intercept

    return volume, slices


def build_affine_from_dicom(slices):
    """Build a 4x4 affine matrix from DICOM orientation/position tags.

    Args:
        slices: Sorted list of pydicom Dataset objects.

    Returns:
        affine: 4x4 numpy array mapping voxel indices to world coordinates (LPS).
    """
    ds = slices[0]
    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
    row_cosine = iop[:3]
    col_cosine = iop[3:]

    ipp_first = np.array([float(x) for x in slices[0].ImagePositionPatient])
    ipp_last = np.array([float(x) for x in slices[-1].ImagePositionPatient])

    ps = [float(x) for x in ds.PixelSpacing]
    n_slices = len(slices)

    if n_slices > 1:
        slice_direction = (ipp_last - ipp_first) / (n_slices - 1)
    else:
        slice_direction = np.cross(row_cosine, col_cosine)
        if hasattr(ds, 'SliceThickness'):
            slice_direction *= float(ds.SliceThickness)

    affine = np.eye(4)
    affine[:3, 0] = row_cosine * ps[1]
    affine[:3, 1] = col_cosine * ps[0]
    affine[:3, 2] = slice_direction
    affine[:3, 3] = ipp_first

    return affine


def dicom_to_nifti(dicom_dir, output_path):
    """Convert a DICOM series to NIfTI format.

    Args:
        dicom_dir: Path to DICOM directory.
        output_path: Output .nii or .nii.gz path.
    """
    import nibabel as nib

    volume, slices = load_dicom_series(dicom_dir)
    affine = build_affine_from_dicom(slices)

    # DICOM uses LPS, NIfTI convention is RAS — flip L→R and P→A
    lps_to_ras = np.diag([-1, -1, 1, 1])
    ras_affine = lps_to_ras @ affine

    img = nib.Nifti1Image(volume, ras_affine)
    nib.save(img, str(output_path))
    print(f"Saved NIfTI: {output_path} — shape {volume.shape}")
