import datetime
import pathlib
import pprint
import re
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import ome_types
import palom
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.segmentation
import tifffile
import zarr
from loguru import logger

PYRAMID_DEFAULTS = dict(
    downscale_factor=2,
    compression="zlib",
    tile_size=1024,
    save_RAM=True,
    is_mask=False,
)

INTENSITY_RESCALE_MAX = 1023
INTENSITY_THRESHOLD_P = 25

RM_SMALL_OBJ_FACTOR = 4

PLOT_PAD_FACTOR = 0.01
PLOT_ASPECT_RATIO_THRESHOLD = 1.2
PLOT_FIGSIZE_SCALE = 1.5
PLOT_DPI = 144


def src_tif_tags(img_path):
    kwargs_tifffile = {}
    try:
        with tifffile.TiffFile(img_path) as tif:
            if tif.pages:
                kwargs_tifffile.update(
                    dict(
                        photometric=tif.pages[0].photometric.value,
                        resolution=tif.pages[0].resolution,
                        resolutionunit=tif.pages[0].resolutionunit.value,
                        software=tif.pages[0].software,
                    )
                )
            else:
                logger.warning(
                    f"No pages found in {img_path}. Returning empty TIFF tags."
                )
    except FileNotFoundError:
        logger.warning(f"Image file not found: {img_path}. Returning empty TIFF tags.")
    except IndexError:
        logger.warning(
            f"Could not access TIFF page 0 in {img_path}. Returning empty TIFF tags."
        )
    except Exception as e:
        logger.warning(
            f"An unexpected error occurred while reading TIFF tags from {img_path}: {e}"
        )
    return kwargs_tifffile


def get_img_path(img_path):
    img_path = pathlib.Path(img_path)
    assert img_path.exists()
    assert re.search(r"(?i).ome.tiff?$", img_path.name) is not None, img_path
    return img_path


def get_output_path(output_path, img_path):
    img_name = re.sub(r"(?i).ome.tiff?$", "", img_path.name)
    out_name = f"{img_name}-bg_compressed.ome.tif"
    if output_path is None:
        output_path = img_path.parent / out_name
    output_path = pathlib.Path(output_path)
    if output_path.is_dir():
        output_path = output_path / out_name
    assert output_path != img_path
    assert re.search(r"(?i).ome.tiff?$", output_path.name) is not None
    output_path.parent.mkdir(exist_ok=True, parents=True)
    return output_path


def local_entropy(img, kernel_size=5):
    img = skimage.exposure.rescale_intensity(
        img, out_range=(0, INTENSITY_RESCALE_MAX)
    ).astype(np.uint16)
    return skimage.filters.rank.entropy(img, np.ones((kernel_size, kernel_size)))


def make_tissue_mask(
    img_path: str | pathlib.Path,
    channel: int,
    thumbnail_level: int,
    img_pyramid_downscale_factor: int,
    entropy_kernel_size: int,
    dilation_radius: int,
    plot: bool = False,
    level_center: float = 0.5,
    level_adjust: int = 0,
):
    """Generates a tissue mask for an OME-TIFF image using entropy-based thresholding and morphological operations.

    Args:
        img_path (str | pathlib.Path): Path to the input OME-TIFF image file.
        channel (int): The channel to use for tissue mask generation.
        thumbnail_level (int): Downsampling level for generating the thumbnail image used for mask generation.
        img_pyramid_downscale_factor (int): Downscaling factor for the pyramid levels in the input image.
        entropy_kernel_size (int): Kernel size of the entropy filter.
        dilation_radius (int): Radius (in pixels at the thumbnail level) for morphological dilation to refine the tissue mask.
        plot (bool, optional): If True, generates a plot of the tissue mask. Defaults to False.
        level_center (float, optional): A normalized adjustment to the thresholding level for tissue mask generation.
            The effective range of this value is dynamically clipped based on the image's entropy characteristics.
            A practical input range is typically between -1.0 and 1.0. Defaults to 0.5.
        level_adjust (int, optional): Adjustment index for threshold levels (range: -2 to 2). Defaults to 0.

    Returns:
        np.ndarray: A boolean NumPy array representing the generated tissue mask at the thumbnail level.

    Notes:
        - The `level_center` parameter is dynamically clipped internally, so its effective range may vary.
    """
    assert thumbnail_level >= 0
    assert img_pyramid_downscale_factor >= 1
    assert entropy_kernel_size > 0
    assert dilation_radius >= 0

    assert level_adjust in np.arange(-2, 3, 1)

    reader = palom.reader.OmePyramidReader(img_path)
    LEVEL = min(thumbnail_level, len(reader.pyramid) - 1)
    _thumbnail = reader.pyramid[LEVEL][channel]
    d_factor = img_pyramid_downscale_factor ** (thumbnail_level - LEVEL)
    thumbnail = np.array(_thumbnail[::d_factor, ::d_factor])

    # thumbnail = np.log1p(thumbnail.astype("float").sum(axis=0))
    # # thumbnail = np.log1p(thumbnail.max(axis=0))
    # entropy_thumbnail = local_entropy(thumbnail, kernel_size=5)

    _tt = np.array(thumbnail)
    np.clip(_tt, np.percentile(_tt[_tt > 0], 0).astype(_tt.dtype), None, out=_tt)
    entropy_thumbnail = local_entropy(np.log1p(_tt), kernel_size=entropy_kernel_size)
    thumbnail = np.log1p(thumbnail)

    erange = np.ptp(entropy_thumbnail)
    _threshold = skimage.filters.threshold_otsu(entropy_thumbnail)
    _max = entropy_thumbnail.max() - 0.1 * erange
    _min = entropy_thumbnail.min() + 0.1 * erange
    _threshold = np.clip(_threshold, _min, _max)

    level_center = np.clip(
        level_center, (_min - _threshold) / erange, (_max - _threshold) / erange
    )

    # forcing threshold to be within 10th and 90th percent of the range
    _threshold += level_center * erange

    thresholds = np.concatenate(
        [
            np.linspace(entropy_thumbnail.min(), _threshold, 4)[1:],
            np.linspace(_threshold, entropy_thumbnail.max(), 4)[1:-1],
        ]
    )
    level_adjusts = np.arange(-2, 3, 1)
    masks = entropy_img_to_masks(
        thumbnail, entropy_thumbnail, thresholds, dilation_radius
    )
    mask = masks[list(level_adjusts).index(level_adjust)]

    if plot:
        fig = plot_tissue_mask(
            thumbnail,
            entropy_thumbnail,
            masks,
            thresholds,
            list(level_adjusts).index(level_adjust),
        )
        fig.suptitle(reader.path.name)

    return mask


def entropy_img_to_masks(
    img, entropy_img, thresholds, dilation_radius, img_is_dark_background=True
):
    if not img_is_dark_background:
        img = -1.0 * img
    masks = np.full((len(thresholds), *entropy_img.shape), fill_value=False, dtype=bool)
    footprint = skimage.morphology.disk(radius=dilation_radius)
    for idx, tt in enumerate(thresholds):
        mask = entropy_img > tt
        mask |= img >= np.percentile(img[mask], INTENSITY_THRESHOLD_P)
        skimage.morphology.binary_dilation(mask, footprint=footprint, out=mask)
        scipy.ndimage.binary_fill_holes(mask, output=mask)
        skimage.morphology.remove_small_objects(
            mask, RM_SMALL_OBJ_FACTOR * footprint.sum(), out=mask
        )

        masks[idx] = mask
    return masks


def plot_tissue_mask(img, entropy_img, masks, thresholds, selected_mask_idx):
    # set contrast min to min value that is not 0
    vimg = skimage.exposure.rescale_intensity(
        img,
        in_range=(img[img > 0].min(), img.max()),
        out_range="float",
    )
    # pad images for mask outline drawing
    pad_size = np.ceil(np.max(img.shape) * PLOT_PAD_FACTOR).astype("int")
    vimg = np.pad(vimg, pad_size, constant_values=0)
    ventropy = np.pad(entropy_img, pad_size, constant_values=entropy_img.min())
    vmasks = np.pad(
        masks,
        [(0, 0), (pad_size, pad_size), (pad_size, pad_size)],
        constant_values=False,
    )
    vmask = vmasks[selected_mask_idx]

    subplot_shape = (2, 1)
    if np.divide(*img.shape) > PLOT_ASPECT_RATIO_THRESHOLD:
        subplot_shape = (1, 2)

    fig, axs = plt.subplots(*subplot_shape, sharex=True, sharey=True)

    axs[0].imshow(vimg, cmap="cividis")
    axs[0].contour(vmask, levels=[0.5], colors=["w"], linewidths=1)
    # axs[1].imshow(ventropy, cmap="cividis", interpolation="none")

    _plot_entropy_mask_levels(
        ventropy, vmasks, thresholds, selected_mask_idx, img=vimg, ax=axs[1]
    )
    return fig


def _plot_entropy_mask_levels(
    entropy_img, masks, thresholds, selected_mask_idx=None, img=None, ax=None
):
    import itertools

    import matplotlib.cm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    assert len(masks) == len(thresholds) == 5

    levels = np.arange(7) - 0.5
    tick_labels = np.arange(5) - 2

    if selected_mask_idx is None:
        selected_mask_idx = 2
    assert selected_mask_idx in range(5)

    if ax is None:
        _, ax = plt.subplots()
    fig = ax.get_figure()

    if img is None:
        img = [[0]]
    ax.imshow(np.log1p(img), cmap="gray")

    ax.contourf(masks.sum(axis=0), cmap="coolwarm_r", levels=levels, alpha=0.75)
    ax.contour(masks[selected_mask_idx], levels=[0.5], colors=["w"], linewidths=1)
    axins = inset_axes(
        ax,
        width=0.1,  # width: .1 inch
        height="100%",  # height: 100%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colors = matplotlib.cm.coolwarm_r(np.linspace(0, 1, 6), alpha=0.75)
    yys = [entropy_img.min()] + list(thresholds) + [entropy_img.max()]

    for pp, cc in zip(itertools.pairwise(yys), colors):
        axins.fill_between([0, 1], *pp, color=cc, step="post")

    axins.set_xlim(0, 1)
    axins.set_ylim(yys[0], yys[-1])
    axins.axes.yaxis.tick_right()
    axins.set_yticks(yys[1:-1], labels=tick_labels)
    axins.set_xticks([])
    axins.axhline(thresholds[selected_mask_idx], color="w", linewidth=3)

    return fig


def write_masked(img_path, output_path, tissue_mask, mask_upscale_factor):
    reader = palom.reader.OmePyramidReader(img_path)

    # match mask shape to full res image
    _, H, W = reader.pyramid[0].shape
    mask_full_zarr = zarr.full((H, W), fill_value=False, chunks=1024)

    mask_full = palom.img_util.repeat_2d(tissue_mask, (mask_upscale_factor,) * 2)[
        :H, :W
    ]
    h, w = mask_full.shape
    # mask_full size might be smaller than image size
    mask_full_zarr[:h, :w] = mask_full[:h, :w]
    mask_full = None

    mosaic = (reader.pyramid[0] * mask_full_zarr).astype(reader.pixel_dtype)

    tif_tags = src_tif_tags(img_path)

    software = "compress_bg_v0"
    tif_tags["software"] = f"{tif_tags.get('software', None)}-{software}"

    palom.pyramid.write_pyramid(
        mosaics=[mosaic],
        output_path=output_path,
        **{
            **dict(
                pixel_size=reader.pixel_size,
                kwargs_tifffile=tif_tags,
            ),
            **PYRAMID_DEFAULTS,
        },
    )

    # FIXME: this may cause mismatch when pyramid configs are different
    ome = ome_types.from_tiff(img_path)
    ome.creator = f"{ome.creator}-{software}"
    tifffile.tiffcomment(output_path, ome.to_xml().encode())

    return output_path


def process_file(
    img_path: str,
    channel: int = 0,
    output_path: str = None,
    only_preview: bool = True,
    thumbnail_level: int = 6,
    img_pyramid_downscale_factor: int = 2,
    entropy_kernel_size: int = 5,
    dilation_radius: int = 5,
    level_center: float = 0.5,
    level_adjust: int = 0,
    skip_qc_plot: bool = False,
):
    """Processes a single OME-TIFF image file, generating a tissue mask, an optional preview, and
    an optionally background-masked OME-TIFF output.

    Args:
        img_path (str): Path to the input OME-TIFF image file.
        channel (int): The channel to use for tissue mask generation. Defaults to 0.
        output_path (str, optional): Path to the output OME-TIFF file. If None, it defaults to
            a file in the same directory as the input image, with "-bg_compressed.ome.tif" suffix.
            If a directory is provided, the output file will be written into it. Defaults to None.
        only_preview (bool, optional): If True, only a tissue mask preview (JPEG) is generated
            without saving the processed OME-TIFF image. Defaults to True.
        thumbnail_level (int, optional): Downsampling level for generating the tissue mask.
            Higher values downsample the image more. Defaults to 6.
        img_pyramid_downscale_factor (int, optional): Downscaling factor for the pyramid levels
            in the input image. Defaults to 2.
        entropy_kernel_size (int, optional): Kernel size (in pixels at the thumbnail level) of the
            entropy filter. Defaults to 5.
        dilation_radius (int, optional): Radius (in pixels at the thumbnail level) for
            morphological dilation to refine the tissue mask. Defaults to 5.
        level_center (float, optional): A normalized adjustment to the thresholding level for
            tissue mask generation. A practical input range is typically between -1.0 and 1.0.
            Defaults to 0.5.
        level_adjust (int, optional): Adjustment index for threshold levels (range: -2 to 2).
            Defaults to 0.
        skip_qc_plot (bool, optional): If True, do not generate and save a tissue mask preview
            (JPEG) for verification. Defaults to False.
    """
    _args = {k: v for k, v in locals().items() if k != "img_path"}

    img_path = get_img_path(img_path)
    output_path = get_output_path(output_path, img_path)

    plot_preview = not skip_qc_plot

    log_path = output_path.parent / "compress-bg-log" / f"{output_path.name}.log"
    log_path.parent.mkdir(exist_ok=True, parents=True)

    handler_id = logger.add(log_path, rotation="5 MB")
    try:
        logger.info(f"Start processing {img_path.name}")
        logger.info(
            f"\nFunction args\n{pprint.pformat(_args, indent=4, sort_dicts=False, width=600)}\n"
        )

        start_time = int(time.perf_counter())

        tissue_mask = make_tissue_mask(
            img_path,
            channel,
            thumbnail_level,
            img_pyramid_downscale_factor,
            entropy_kernel_size,
            dilation_radius,
            plot=plot_preview,
            level_center=level_center,
            level_adjust=level_adjust,
        )
        # skip masking when tissue area > 90%
        if tissue_mask.sum() / tissue_mask.size > 0.9:
            tissue_mask[:] = True

        if plot_preview:
            fig = plt.gcf()
            fig.set_size_inches(fig.get_size_inches() * PLOT_FIGSIZE_SCALE)
            fig.savefig(
                output_path.parent
                / re.sub(r"(?i).ome.tiff?$", ".jpg", output_path.name),
                bbox_inches="tight",
                dpi=PLOT_DPI,
            )
            plt.close(fig)

        if not only_preview:
            write_masked(
                img_path,
                output_path,
                tissue_mask,
                img_pyramid_downscale_factor**thumbnail_level,
            )

        end_time = int(time.perf_counter())
        logger.info(
            f"Done processing (elapsed {datetime.timedelta(seconds=end_time - start_time)}) {img_path.name}\n"
        )
    finally:
        logger.remove(handler_id)


def run_batch(csv_path, print_args=True, dryrun=False, **kwargs):
    """Processes multiple OME-TIFF image files in batch mode using parameters provided in a CSV file.

    This function reads a CSV file where each row represents a processing job and columns
    correspond to arguments of the `process_file` function. Command-line arguments provided
    to `run_batch` will be overridden by values specified in the CSV for the same parameter.

    Args:
        csv_path (str): Path to the CSV file containing batch processing instructions.
            The CSV must include an 'img_path' column and can optionally include other
            parameters for `process_file`.
        print_args (bool, optional): If True, prints the function arguments for `process_file`
            before processing. Defaults to True.
        dryrun (bool, optional): If True, prints the arguments for each job in the batch
            without executing them, allowing for verification of parameters. Defaults to False.
        **kwargs: Additional keyword arguments passed directly to the `process_file` function.
            These arguments will be overridden by values from the CSV if present.
    """
    import csv
    import inspect
    import pprint
    import types

    from fire.parser import DefaultParseValue

    func = process_file

    if print_args:
        _args = [str(vv) for vv in inspect.signature(func).parameters.values()]
        print(f"\nFunction args and defaults\n{pprint.pformat(_args, indent=4)}\n")
    _arg_types = inspect.get_annotations(func)
    arg_types = {}
    for k, v in _arg_types.items():
        if isinstance(v, types.UnionType):
            v = v.__args__[0]
        arg_types[k] = v

    csv_kwargs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row_idx, rr in enumerate(reader, start=1):
            try:
                kk_row = {}
                for kk, vv in rr.items():
                    if (kk in arg_types) and (vv is not None) and (len(str(vv).strip()) > 0):
                        try:
                            val = DefaultParseValue(vv)
                            kk_row[kk] = arg_types[kk](val)
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Row {row_idx}: Could not convert {kk}='{vv}' to {arg_types[kk].__name__}. Error: {e}. Skipping row."
                            )
                            raise ValueError("Skip row") from e
                csv_kwargs.append(kk_row)
            except ValueError:
                continue

    if dryrun:
        for kk in csv_kwargs:
            pprint.pprint({**kwargs, **kk}, sort_dicts=False)
            print()
        return

    fail_count = 0
    for kk in csv_kwargs:
        try:
            func(**{**kwargs, **kk})
        except Exception as e:
            img_path = kk.get("img_path", "unknown")
            logger.error(f"Failed to process {img_path}\n{e}\n{traceback.format_exc()}")
            fail_count += 1

    if fail_count > 0:
        logger.error(f"Batch processing completed with {fail_count} failures.")
    else:
        logger.info("Batch processing completed successfully.")


def main():
    import sys

    import fire

    logger.remove()
    logger.add(sys.stderr)
    fire.Fire({"run": process_file, "run-batch": run_batch})


if __name__ == "__main__":
    import sys

    sys.exit(main())
