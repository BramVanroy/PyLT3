"""
If you want to replace certain occurrences in a file, for instance because you want to make collocations by
delimiting a keyword by tabs, you can easily do that with this function. In the output file, the keyword will be
surrounded by tabs so that tools such as AntConc can easily recognise it.
"""

from pylt3.utils import file_helpers

import re
from pathlib import Path
import sys


def collocate_out(line, index, word, fout):
    # Skip header line which has index zero
    if index > 0:
        line = line.strip()
        # Surround an input word (or its plural) with tabs
        replaced = re.sub(r"\b(%ss?)\b" % re.escape(word), r"\t\1\t", line, flags=re.IGNORECASE)
        fout.write(replaced + "\n")


if __name__ == "__main__":
    my_file = str(Path(sys.argv[1]))
    my_string = str(sys.argv[2])

    with open("collocation.txt", "w") as f:
        file_helpers.scan_file_and_execute(my_file, lambda line, index: collocate_out(line, index, my_string, f))
