from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("compress-bg")
except PackageNotFoundError:
    # package is not installed
    pass

from .main import main, process_file, run_batch