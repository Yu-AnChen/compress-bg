# Documentation for `compress-bg`

This script processes OME-TIFF image files to create compressed
background-masked versions, using a variety of image processing techniques to
generate tissue masks. It supports individual file processing and batch
processing through CSV input.

## Features

- Image Background Masking: Creates a tissue mask to filter out unwanted areas
  from OME-TIFF images.
- Tissue Mask Generation: Generates masks using local entropy, thresholds, and
  dilation techniques.
- OME-TIFF Metadata Preservation: Retains metadata while saving the processed
  image.
- Batch Processing: Supports batch processing via a CSV file for bulk file
  handling.
- Preview Mode: Optionally generates preview images of tissue masks for
  verification before full processing.

## How to Use

### Process 1 file (`compress-bg.py run`)

To process a single OME-TIFF file, use the `run` sub-command.

#### Command (`compress-bg.py run`)

```bash
python compress-bg.py run --img_path <path-to-input-image> [OPTIONS]
```

#### Parameters (`compress-bg.py run`)

- `img_path` (required): Path to the input OME-TIFF image file.
- `channel` (default: 0): The channel to use for tissue mask generation.
- `output_path` (optional): Path to the output ome-tiff. If not provided, the
  output file will be in the same directory as the input image, the file name
  ends with `-bg_compressed.ome.tif`. If a directory is provided, it must be
  already created, and the output files will be written to it.
- `only_preview` (default: `True`): If `True`, generates a tissue mask preview
  without saving a processed image.
- `thumbnail_level` (default: 6): Downsampling level for generating the tissue
  mask. Higher values downsample the image more.
- `img_pyramid_downscale_factor` (default: 2): Downscaling factor for the
  pyramid levels in the input image.
- `entropy_kernel_size` (default: 5): Kernel size (in pixels at the thumbnail
  level) of the entropy filter.
- `dilation_radius` (default: 5): Radius (in pixels at the thumbnail level) for
  morphological dilation to refine the tissue mask.
- `level_center` (default: 0.5): A normalized adjustment to the thresholding
  level for tissue mask generation. While the internal implementation
  dynamically clips this value based on image characteristics, a practical input
  range for users is typically between -1.0 and 1.0.
- `level_adjust` (default: 0): Adjustment index (see the preview visualization)
  for threshold levels (range: -2 to 2).
- `skip_qc_plot` (default: `False`): If `True`, do not generate and save a
  tissue mask preview (JPEG) for verification.

#### Example (`compress-bg.py run`)

```bash
python compress-bg.py run --img_path input.ome.tif --output_path output.ome.tif --only_preview False --thumbnail_level 7
```

### Batch Processing (`compress-bg.py run-batch`)

To process multiple files, use the `run-batch` sub-command with a CSV input
file.

#### Command (`compress-bg.py run-batch`)

```bash
python compress-bg.py run-batch --csv_path <path-to-csv> [OPTIONS]
```

#### CSV Format (`compress-bg.py run-batch`)

At a minimum, the CSV file must include an **img_path** column. Additional
columns corresponding to the options available in the single-file processing
command (`compress-bg.py run`), such as **thumbnail_level**,
**dilation_radius**, and others, can be included to customize the values for
those options. Each row in the CSV represents a single processing job.

Example CSV (files-minimal.csv):

```csv
img_path
/path/to/image1.ome.tif
/path/to/image2.ome.tif
```

Example CSV (files-extensive.csv):

```csv
img_path,output_path,only_preview,thumbnail_level,img_pyramid_downscale_factor,dilation_radius,level_center,level_adjust
/path/to/image1.ome.tif,/path/to/output1.ome.tif,False,6,2,2,-0.2,0
/path/to/image2.ome.tif,/path/to/output2.ome.tif,True,5,2,2,0.1,-1
```

#### Parameters (`compress-bg.py run-batch`)

- `csv_path` (required): Path to the CSV file containing batch processing
  instructions.
- `print_args` (default: True): Prints the function arguments.
- `dryrun` (default: False): If True, prints the arguments for each job in the
  batch without executing them.
- Other flags (optional): other flags in the `run` command can be specified,
  such as `--level_center 0.2`. Note that if the same flag is also specified in
  the CSV column, the value in the CSV file will override

#### Example (`compress-bg.py run-batch`)

```bash
python compress-bg.py run-batch --csv_path files-minimal.csv --output_path /path/to/output/directory --thumbnail_level 8
```

### Generated Outputs

1. Tissue Mask Preview (JPEG): A visual representation of the tissue mask
   applied to the image. Saved in the same directory as the output file, with
   the .jpg extension.
1. Processed Image (OME-TIFF): A compressed, background-masked version of the
   input image. Metadata is preserved and updated.
1. Log File: A log file is generated in the compress-bg-log folder within the
   output directory.
