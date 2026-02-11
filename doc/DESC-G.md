# Algorithmic Methodology for Background Masking in Large-Scale Histology Images

## Abstract
This document describes the algorithmic workflow implemented in `compress-bg` for the automated detection and masking of tissue regions in high-resolution OME-TIFF images. The primary objective is to facilitate data compression by zeroing out non-informative background regions while preserving tissue architecture through a combination of entropy-based segmentation and morphological refinement.

## 1. Multiscale Data Handling and Pre-processing
The algorithm utilizes a pyramidal data structure to manage high-bit-depth, large-scale microscopy images. 
- **Thumbnail Extraction**: To achieve computational efficiency, a low-resolution representation (thumbnail) is extracted from the OME-Pyramid.
- **Normalization**: The thumbnail intensity $I$ is subject to a log-transform $L = \log(1 + I)$ to stabilize variance and enhance low-intensity signals, followed by a linear rescaling to a 10-bit range (0–1023) to satisfy the input requirements of rank-based filters.

## 2. Entropy-based Tissue Detection
Tissue regions often exhibit higher structural complexity compared to the stochastic noise of the background. This complexity is quantified using local entropy.
- **Local Entropy Calculation**: A local entropy filter is applied to the log-transformed thumbnail using a square kernel of size $k 	imes k$ (default $k=5$). The entropy $H$ at each pixel is calculated based on the local grayscale distribution.
- **Adaptive Thresholding**: An initial threshold $T_{Otsu}$ is determined using Otsu's method on the entropy map. To provide robustness against varying signal-to-noise ratios, the threshold is dynamically adjusted:
  $$T_{final} = 	ext{clip}(T_{Otsu}, 	ext{min}, 	ext{max}) + (	ext{level\_center} \cdot \Delta H)$$
  where $\Delta H$ is the peak-to-peak range of the entropy map, and $	ext{level\_center}$ is a tunable parameter (default 0.5) for sensitivity adjustment.

## 3. Mask Refinement and Morphological Processing
The raw entropy mask is further refined to ensure contiguous tissue coverage and inclusion of low-entropy but high-intensity features.
- **Intensity Integration**: Pixels exceeding a specific percentile (default 25th) of the intensity within the initial entropy mask are included to ensure faint tissue regions are not discarded.
- **Morphological Dilation**: A circular footprint of radius $r$ is used for dilation to bridge small gaps and create a safety margin around tissue boundaries.
- **Hole Filling**: Topological holes within the tissue mask are closed to maintain the integrity of internal structures.
- **Small Object Removal**: Spurious detections are eliminated by removing connected components with an area smaller than a threshold proportional to the dilation footprint (factor 4).

## 4. Mask Application and Pyramid Reconstruction
The resulting binary mask $M_{low}$ is upsampled to the full resolution $M_{high}$ using nearest-neighbor interpolation.
- **Background Suppression**: The full-resolution image $J$ is masked by the bitwise multiplication $J_{masked} = J \odot M_{high}$, effectively setting background pixels to zero.
- **Output Encoding**: The masked data is streamed using Dask to manage memory constraints and rewritten into a tiled OME-TIFF pyramid with Zlib compression. This process yields significant storage savings as the zero-valued background tiles compress highly efficiently.

## 5. Software Stack
The implementation leverages several key scientific Python libraries:
- **Dask / Zarr**: For lazy evaluation and chunked processing of arrays exceeding system RAM.
- **Palom**: For handling OME-TIFF pyramidal structures and metadata preservation.
- **Scikit-image**: For rank-based entropy filtering, thresholding, and morphological operations.
- **Scipy**: For topological hole filling.
- **Matplotlib**: For generating quality control (QC) visualizations overlaying the computed mask on the original intensity data.
