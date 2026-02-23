# Data

## AutoPET Dataset (TCIA)

The primary dataset used in these tutorials is AutoPET from The Cancer Imaging Archive (TCIA).

### Download Instructions

1. Visit [TCIA AutoPET](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287)
2. Install the NBIA Data Retriever
3. Download the manifest file and load it in the retriever
4. Extract to `data/AutoPET/`

### Expected Structure

```
data/AutoPET/
  ├── FDG-PET-CT-Lesions/
  │   ├── PETCT_0000/
  │   │   ├── CT/
  │   │   ├── PET/
  │   │   └── SEG/
  │   └── ...
```

## Synthetic Fallbacks

Every notebook includes synthetic data generators so you can run the full pipeline without downloading clinical data. The synthetic generators create:

- **Shepp-Logan phantom** — standard test image for reconstruction algorithms
- **Synthetic PET slices** — Gaussian blobs simulating radiotracer uptake
- **Paired non-AC/AC data** — simulated attenuation effects for GAN training
- **Synthetic patient cohorts** — for statistical validation demos

These are sufficient for learning the concepts, though results will differ from real clinical data.
