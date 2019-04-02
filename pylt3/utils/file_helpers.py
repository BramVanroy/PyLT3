import linecache
from pathlib import Path
import locale
from math import floor

from sklearn.model_selection import train_test_split

from .type_helpers import verify_kwargs, is_simple_list


def get_number_of_lines(fin):
    pfin = Path(fin).resolve()

    i = 0
    with open(str(pfin)) as fhin:
        for i, _ in enumerate(fhin, 1):
            pass

    return i

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

    for entry in Path(root).glob('*'):
        if entry.is_dir() and entry.name not in exclude_dirs:
            if kwargs['recursive']:
                # If truth-y value: keep value, otherwise use None
                next_exclude = exclude_dirs if exclude_dirs else None
                scan_dir_and_execute(entry.path, exec_func, next_exclude, verbose=verbose, recursive=True)
        elif entry.is_file():
            if verbose > 1:
                print(f"\tProcessing {entry.name}", flush=True)

            exec_func(str(entry))

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
            line_i += 1

    return None


def concatenate_files(input_item, output_file, extension=None, separator=None, remove_headers=0, verbose=0, **kwargs):
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
        if not extension or str(file_path).endswith(extension):
            files_concat_n = files_concat_n+1
            with open(file_path, 'r', encoding=kwargs['encoding']) as fin:
                line_n = 0
                for line in fin:
                    line_n = line_n+1
                    if (files_concat_n == 1 and kwargs['retain_first_header']) or line_n > remove_headers:
                        _fout.write(line)

            if separator:
                _fout.write(str(separator))
        else:
            files_skipped_n += 1

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
        for key, val in simple_dict.items():
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


def _get_split_idxs(data_size, shuffle, train_size, test_size, dev_size):
    indxs = list(range(data_size))

    train_dev_size = train_size + dev_size
    train_idxs, test_idxs = train_test_split(indxs, train_size=train_dev_size, test_size=test_size, shuffle=shuffle)

    if dev_size is not None:
        train_idxs, dev_idxs = train_test_split(train_idxs, train_size=train_size, test_size=dev_size, shuffle=shuffle)
    else:
        dev_idxs = None

    return train_idxs, test_idxs, dev_idxs


def split_files(input_files, train_size, test_size, dev_size=None, output_exts=None, shuffle=False, output_dir=False):
    split_sizes = list(filter(None, [train_size, test_size, dev_size]))

    if output_exts is None:
        output_exts = ['train', 'test'] if dev_size is None else ['train', 'test', 'dev']
    elif len(split_sizes) != len(output_exts):
        raise ValueError("The size of output_exts has to be the same as the number of sizes.")

    # Get number of lines of input
    input_paths = [Path(f).resolve() for f in input_files]
    nro_lines = [get_number_of_lines(p) for p in input_paths]

    # All input files must be the same size
    if len(set(nro_lines)) > 1:
        raise ValueError(f"The number of lines is not the same for your input files. The sizes are: {nro_lines}.")
    else:
        nro_lines = nro_lines[0]

    # if all sizes are less than (or equal to) 1, then we're probably using percentages
    percent_divide = all([True if x <= 1 else False for x in split_sizes])

    # Get missing values. One of the given values can be -1 and its actual value will be calculated.
    if split_sizes.count(-1) > 1:
        raise ValueError("Only one size can be -1.")
    elif split_sizes.count(-1) == 1:
        unk_idx = split_sizes.index(-1)

        # +1 to make up for the current -1
        if percent_divide:
            split_sizes[unk_idx] = 1 - sum(split_sizes) - 1
        else:
            split_sizes[unk_idx] = nro_lines - sum(split_sizes) - 1

    if percent_divide and sum(split_sizes) != 1:
        raise ValueError("When using percentages, the split has to sum up to 1.")
    elif not percent_divide and sum(split_sizes) != nro_lines:
        raise ValueError("When using absolute numbers, the split has to sum up to the total number of lines."
                         f" Got {sum(split_sizes)}, expected {nro_lines}.")

    if percent_divide:
        # Gets absolute quantities
        train_size = floor(nro_lines * split_sizes[0])
        if dev_size is None:
            test_size = nro_lines - train_size
        else:
            test_size = floor(nro_lines * split_sizes[1])
            dev_size = nro_lines - train_size - test_size

        split_sizes = list(filter(None, [train_size, test_size, dev_size]))

        if sum(split_sizes) != nro_lines:
            raise ValueError(f"Shape mismatch. The total sum of splits is {sum(split_sizes)} but expected {nro_lines}.")

    train_idxs, test_idxs, dev_idxs = _get_split_idxs(nro_lines, shuffle, *split_sizes)
    print(sorted(train_idxs))
    print(sorted(test_idxs))
    print(sorted(dev_idxs))
    idxs = {'train': train_idxs, 'test': test_idxs, 'dev': dev_idxs}

    # Look up the relevant lines, and write to file.
    for pin in input_paths:
        for ext_id, partition in enumerate(['train', 'test', 'dev']):
            # Continue if dev is None
            if idxs[partition] is None:
                continue

            pout = pin.parent

            if output_dir:
                pout = pout.joinpath(output_exts[ext_id])
                pout.mkdir(exist_ok=True)

            pout = pout.joinpath(pin.name + '.' + output_exts[ext_id])
            with open(str(pout), 'w', encoding='utf-8') as fhout:
                for line_id in idxs[partition]:
                    fhout.write(linecache.getline(str(pin), line_id))
