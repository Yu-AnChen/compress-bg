# TODO for `compress-bg.py`

## `README.md` Issues

1.  **`level_center` Range Inconsistency**: The `README` states that `level_center` "Should be between -1 and 1", but the `make_tissue_mask` function in the script has a commented-out assertion for `(-0.5, 0.5)` and the active code dynamically clips the value based on image entropy. Please clarify the effective or recommended range for this parameter in the documentation.
2.  **Typo in Batch Command Example**: The example command for batch processing uses `python script.py run-batch` instead of `python compress-bg.py run-batch`.

## `compress-bg.py` Issues and Suggestions

1.  **Missing Docstrings**: Several key functions, notably `make_tissue_mask`, `process_file`, and `run_batch`, lack comprehensive docstrings. Adding them would greatly improve code readability and maintainability.
2.  **Broad Exception Handling**:
    *   In `write_masked`, the `try...except Exception:` block around `src_tif_tags` is too broad. It's better to catch specific exceptions (e.g., `IndexError` if `tif.pages[0]` fails, `FileNotFoundError`).
    *   The `src_tif_tags` function itself could be made more robust by checking if `tif.pages` is not empty before attempting to access `tif.pages[0]`.
3.  **`get_output_path` Redundancy**: The line `output_path = pathlib.Path(output_path)` is called twice if `output_path` is provided as an argument. It can be moved outside the `if output_path is None:` block to avoid this.
4.  **`make_tissue_mask` Refinements**:
    *   **Simplified `LEVEL` Calculation**: The `LEVEL` adjustment can be simplified from:
        ```python
        LEVEL = thumbnail_level
        if LEVEL > len(reader.pyramid) - 1:
            LEVEL = len(reader.pyramid) - 1
        ```
        to `LEVEL = min(thumbnail_level, len(reader.pyramid) - 1)`.
    *   **Commented-out Code**: The commented-out `palom.img_util.cv2_downscale_local_mean` suggests an alternative. If it's not being used, it's best to remove it or add a comment explaining why it's there.
    *   **Magic Number `[:2]`**: The `_thumbnail = reader.pyramid[LEVEL][:2]` line processes only the first two channels. This should be explained or parameterized if the script might handle images with different channel counts.
5.  **`entropy_img_to_masks` Parameter Naming**: The `img_is_dark_backgroud` parameter's effect is to negate the image if `False`. A more descriptive name like `invert_image_for_masking` could improve clarity.
6.  **Plotting Logic in `process_file`**:
    *   The `plot=True` argument is hardcoded in the call to `make_tissue_mask` within `process_file`, meaning a plot is always generated regardless of the `only_preview` parameter (which controls saving the *processed image*).
    *   The `matplotlib.pyplot` imports are done inside `process_file` and plotting functions. While acceptable for less frequent use, for a script where plotting is a core feature, moving them to the top level is common practice.
    *   Consider making the plotting conditional on an explicit `plot` parameter in `process_file` if you want more control over when previews are generated.
7.  **Logger Reconfiguration**: In `process_file`, `logger.remove()` and `logger.add(...)` reconfigure the `loguru` loggers every time the function is called. In a batch processing scenario, this can be inefficient and might lead to unexpected behavior if global logging configurations are in place. It's generally better to configure loggers once at the application's entry point (e.g., in `main()`) or manage handlers more granularly.
8.  **`run_batch` Robustness**:
    *   **Type Conversion Error Handling**: The line `arg_types[kk](DefaultParseValue(vv))` attempts to convert string values from the CSV to their respective Python types. If a value in the CSV cannot be converted (e.g., "abc" for an `int` parameter), it will raise an error and halt the batch. Implement `try-except` blocks here to catch conversion errors and provide informative messages, allowing the batch to continue processing other files.
    *   **`fire.parser.DefaultParseValue`**: Relying on internal `fire.parser` details might be brittle. Consider a more explicit way to cast types from the CSV or rely on `fire`'s automatic parsing if it handles errors gracefully.
    *   **Error Logging in Batch**: If `func(**{**kwargs, **kk})` fails for a specific file in a batch, the entire batch processing stops. It would be more robust to log the error for that file and continue with the next one, providing a summary of failures at the end.
9.  **Magic Numbers**: Several numeric literals (e.g., `1024-1` in `local_entropy`, `0.01`, `1.2`, `1.5`, `144` in plotting functions, `25` in `entropy_img_to_masks`) could be replaced with named constants for better readability and easier modification.
10. **`process_file` `_args` Modification**: The `del _args["img_path"]` line modifies the `locals()` dictionary directly. While it might not cause issues here, it's generally safer to create a filtered copy of `locals()` if you intend to modify it for logging purposes.
