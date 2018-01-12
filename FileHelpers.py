import os
import locale


def scandir_and_execute(root, exec_func, exclude_dirs=None, recursive=True, verbose=0):
    # Rather than use an empty list that would be mutated at each call (Mutable Default Argument), use None and check
    if exclude_dirs is None:
        exclude_dirs = []

    if not 0 <= verbose <= 2:
        raise ValueError(f'Unexpected value {verbose} for verbose')

    if verbose > 0:
        print(f"TRAVERSING {root}", flush=True)

    for entry in os.scandir(root):
        if entry.is_dir() and entry.name not in exclude_dirs:
            if recursive:
                # If truth-y value: keep value, otherwise convert to None
                next_exclude = exclude_dirs if exclude_dirs else None
                scandir_and_execute(entry.path, exec_func, next_exclude, True, verbose)
        elif entry.is_file():
            if verbose > 1:
                print(f"\tProcessing {entry.name}", flush=True)

            exec_func(entry.path)


def scanfile_and_execute(file, exec_func, encoding=locale.getpreferredencoding(), remove_nl=True, verbose=False):
    if verbose:
        print(f"READING {file}", flush=True)

    line_n = 0
    # newLine=None standardises the new line character to \n
    with open(file, encoding=encoding, newline=None) as f:
        for line in f:
            line_n = line_n+1
            if verbose:
                print(f"\tProcessing line n. {line_n}", end="\r", flush=True)
            if remove_nl:
                line = line.rstrip('\n')

            exec_func(line)

        if verbose:
            print("")
