import os
import locale


def scandir_and_execute(root, exec_func, exclude_dirs=None, recursive=True, verbose=0):
    if verbose not in range(0, 3):
        raise ValueError(f'Unexpected value {verbose} for verbose')

    # Rather than use an empty list that would be mutated at each call (Mutable Default Argument), use None and check
    if exclude_dirs is None:
        exclude_dirs = []

    if verbose > 0:
        print(f"TRAVERSING {root}", flush=True)

    for entry in os.scandir(root):
        if entry.is_dir() and entry.name not in exclude_dirs:
            if recursive:
                # If truth-y value: keep value, otherwise use None
                next_exclude = exclude_dirs if exclude_dirs else None
                scandir_and_execute(entry.path, exec_func, next_exclude, True, verbose)
        elif entry.is_file():
            if verbose > 1:
                print(f"\tProcessing {entry.name}", flush=True)

            exec_func(entry.path)


def scanfile_and_execute(file, exec_func, encoding=locale.getpreferredencoding(), verbose=0):
    if verbose not in range(0, 3):
        raise ValueError(f'Unexpected value {verbose} for verbose')

    line_n = 0
    with open(file, encoding=encoding) as f:
        for line in f:
            if verbose > 0:
                line_n = line_n+1
                proc_str = "Processing"
                if verbose > 1:
                    proc_str += f" file {file}"
                proc_str += f" line n. {line_n}"
                print(proc_str, end="\r", flush=True)

            exec_func(line)

        if verbose > 0:
            print("", flush=True)
