import os
import locale
import operator


def scan_dir_and_execute(root, exec_func, exclude_dirs=None, recursive=True, verbose=0):
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
                scan_dir_and_execute(entry.path, exec_func, next_exclude, True, verbose)
        elif entry.is_file():
            if verbose > 1:
                print(f"\tProcessing {entry.name}", flush=True)

            exec_func(entry.path)

    return None


def scan_file_and_execute(file, exec_func, encoding=locale.getpreferredencoding(), verbose=0):
    if verbose not in range(0, 3):
        raise ValueError(f'Unexpected value {verbose} for verbose')

    line_i = 0
    with open(file, encoding=encoding) as f:
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


def concatenate_files(input_item, output_file, extension=None, remove_headers=0, retain_first_header=False,
                      recursive=True, encoding=locale.getpreferredencoding(), verbose=0):
    if verbose not in range(0, 3):
        raise ValueError(f'Unexpected value {verbose} for verbose')

    if remove_headers and not remove_headers > 0:
        raise ValueError(f'Unexpected value {remove_headers} for remove_headers. Use a positive integer that indicates '
                         f'how many lines should be removed from the top of each file. True will remove the first line')

    if not isinstance(retain_first_header, bool):
        raise ValueError(f'Unexpected value {retain_first_header} for retain_first_header. A boolean value is required')

    # If the input is a list (not a str instance), set switch to True
    is_file_list = False if isinstance(input_item, str) else True

    extension = '' if not extension else extension

    files_skipped_n = 0
    files_concat_n = 0

    def append_to_file(file_path, _fout, ext):
        nonlocal files_concat_n, files_skipped_n
        if not ext or file_path.endswith(f".{ext}"):
            files_concat_n = files_concat_n+1
            with open(file_path, 'r') as fin:
                line_n = 0
                for line in fin:
                    line_n = line_n+1
                    if (files_concat_n == 1 and retain_first_header) or line_n > remove_headers:
                        _fout.write(line)
        else:
            files_skipped_n = files_skipped_n+1

        return None

    with open(output_file, 'w', encoding=encoding) as fout:
        if is_file_list:
            for file in input_item:
                file = os.path.normpath(file)
                if os.path.isfile(file):
                    append_to_file(os.path.normpath(file), fout, extension)
                else:
                    raise FileNotFoundError(f'File {file} does not exist')
        else:
            scan_dir_and_execute(input_item, lambda _file: append_to_file(_file, fout, extension), recursive=recursive,
                                 verbose=verbose)
    if verbose > 0:
        print(f"Finished! Concatenated {files_concat_n} files, skipped {files_skipped_n} files", flush=True)

    return output_file


def print_simple_dict(simple_dict, output_file, sort_on=None, reverse=False, encoding=locale.getpreferredencoding()):
    if sort_on:
        if sort_on == 'keys':
            print_dict = sorted(simple_dict.items(), key=operator.itemgetter(0), reverse=reverse)
        elif sort_on == 'values':
            print_dict = sorted(simple_dict.items(), key=operator.itemgetter(1), reverse=reverse)
        else:
            raise ValueError(f'Unexpected value {sort_on} for sort_on. Use None, keys, or values')
    else:
        print_dict = list(simple_dict.items())

    with open(output_file, 'w', encoding=encoding) as fout:
        for tupe in print_dict:
                fout.write(f"{tupe[0]}\t{tupe[1]}\n")

    return output_file
