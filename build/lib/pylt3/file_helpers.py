from os import scandir
from pathlib import Path
import locale

from pylt3.type_helpers import verify_kwargs, is_simple_list


def scan_dir_and_execute(root, exec_func, exclude_dirs=None, verbose=0, **kwargs):
    default_params = {'recursive': True}
    kwargs = verify_kwargs(default_params, kwargs)

    if verbose not in range(0, 3):
        raise ValueError(f"Unexpected value {verbose} for verbose. 0, 1, or 2 expected")

    # Rather than use an empty list that would be mutated at each call (Mutable Default Argument), use None and check
    if exclude_dirs is None:
        exclude_dirs = []

    if verbose > 0:
        print(f"TRAVERSING {root}", flush=True)

    for entry in scandir(root):
        if entry.is_dir() and entry.name not in exclude_dirs:
            if kwargs['recursive']:
                # If truth-y value: keep value, otherwise use None
                next_exclude = exclude_dirs if exclude_dirs else None
                scan_dir_and_execute(entry.path, exec_func, next_exclude, verbose=verbose, recursive=True)
        elif entry.is_file():
            if verbose > 1:
                print(f"\tProcessing {entry.name}", flush=True)

            exec_func(entry.path)

    return None


def scan_file_and_execute(file, exec_func, verbose=0, **kwargs):
    default_params = {'encoding': locale.getpreferredencoding()}
    kwargs = verify_kwargs(default_params, kwargs)

    if verbose not in range(0, 3):
        raise ValueError(f"Unexpected value {verbose} for verbose")

    line_i = 0
    with open(file, encoding=kwargs['encoding']) as f:
        for line in f:
            if verbose > 0:
                proc_str = "Processing"
                if verbose > 1:
                    proc_str += f" file {file}"
                proc_str += f" line {line_i+1}"
                print(proc_str, end="\r", flush=True)

            exec_func(line, line_i)
            line_i = line_i + 1

    return None


def concatenate_files(input_item, output_file, extension=None, remove_headers=0, verbose=0, **kwargs):
    default_params = {'encoding': locale.getpreferredencoding(), 'recursive': True, 'retain_first_header': False}
    kwargs = verify_kwargs(default_params, kwargs)

    if verbose not in range(0, 3):
        raise ValueError(f"Unexpected value {kwargs['verbose']} for verbose")

    if remove_headers and not remove_headers > 0:
        raise ValueError(f"Unexpected value {remove_headers} for remove_headers. Use a positive integer that indicates "
                         f"how many lines should be removed from the top of each file. True will remove the first line")

    if extension is None:
        extension = ''
    elif not isinstance(extension, str):
        raise ValueError(f"Unexpected value {extension} for extension. A str value is expected")

    is_file_list = True if isinstance(input_item, list) else False

    files_skipped_n = 0
    files_concat_n = 0

    def append_to_file(file_path, _fout):
        nonlocal files_concat_n, files_skipped_n
        if not extension or file_path.endswith(f".{extension}"):
            files_concat_n = files_concat_n+1
            with open(file_path, 'r') as fin:
                line_n = 0
                for line in fin:
                    line_n = line_n+1
                    if (files_concat_n == 1 and kwargs['retain_first_header']) or line_n > remove_headers:
                        _fout.write(line)
        else:
            files_skipped_n = files_skipped_n+1

        return None

    with open(output_file, 'w', encoding=kwargs['encoding']) as fout:
        if is_file_list:
            for file in input_item:
                # Resolve, i.e. ensure rel->abs path, and append
                append_to_file(Path(file).resolve(), fout)
        else:
            scan_dir_and_execute(input_item, lambda _file: append_to_file(_file, fout), recursive=kwargs['recursive'],
                                 verbose=verbose)
    if verbose > 0:
        print(f"Finished! Concatenated {files_concat_n} files, skipped {files_skipped_n} files", flush=True)

    return output_file


def print_simple_dict(simple_dict, output_file, **kwargs):
    default_params = {'encoding': locale.getpreferredencoding()}
    kwargs = verify_kwargs(default_params, kwargs)

    with open(output_file, 'w', encoding=kwargs['encoding']) as fout:
        for key, val in simple_dict:
            if is_simple_list(key):
                key = "\t".join(key)
            if is_simple_list(val):
                val = "\t".join(val)

            fout.write(f"{key}\t{val}\n")

    return output_file


def print_tuplelist(tupelist, output_file, **kwargs):
    default_params = {'encoding': locale.getpreferredencoding()}
    kwargs = verify_kwargs(default_params, kwargs)

    with open(output_file, 'w', encoding=kwargs['encoding']) as fout:
        for tupe in tupelist:
            key = tupe[0]
            val = tupe[1]
            if is_simple_list(key):
                key = "\t".join(key)
            if is_simple_list(val):
                val = "\t".join(val)

            fout.write(f"{key}\t{val}\n")

    return output_file
