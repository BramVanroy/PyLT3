"""
If you want to execute a function only for files with a given extension, you could do something like this.
"""
from pylt3.utils import file_helpers

from pathlib import Path
import sys


def parse_file(file, extension):
    if file.endswith(f".{extension}"):
        print(file)
        # Do something


if __name__ == "__main__":
    my_dir = str(Path(sys.argv[1]))
    my_ext = sys.argv[2]
    file_helpers.scan_dir_and_execute(my_dir, lambda file: parse_file(file, my_ext))