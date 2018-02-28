from string import punctuation
from operator import itemgetter


def is_simple_list(l):
    return isinstance(l, list) or isinstance(l, tuple)

def clean_simple_dict(simple_dict, side='keys', **kwargs):
    default_params = {'rm_only_punct': False, 'rm_contains_punct': False, 'rm_only_digits': False,
                      'rm_contains_digits': False, 'rm_only_nonan': False, 'rm_contains_nonan': False}
    kwargs = verify_kwargs(default_params, kwargs)

    if not (side == 'keys' or side == 'values'):
        raise ValueError(f"Unexpected value {side} for side. Use keys or values")

    def contains(token, contain_type):
        for char in token:
            if contain_type == 'nonan' and not char.isalnum():
                return True
            elif contain_type == 'digit' and char.isdigit():
                return True
            elif contain_type == 'punct' and char in punctuation:
                return True

        return False

    def only_contains(token, contain_type):
        for char in token:
            if contain_type == 'nonan' and char.isalnum():
                return False
            elif contain_type == 'punct' and char not in punctuation:
                return False

        return True

    clean_dict = {}

    for key, val in simple_dict.items():
        # If key or value is tuple, the first item of the tuple will be used instead
        simple_key = key[0] if is_simple_list(key) else key
        simple_val = val[0] if is_simple_list(val) else val

        if kwargs['rm_only_nonan'] and ((side == 'keys' and only_contains(simple_key, 'nonan')) or
                                        (side == 'values' and only_contains(simple_val, 'nonan'))):
            continue
        elif kwargs['rm_only_punct'] and ((side == 'keys' and only_contains(simple_key, 'punct')) or
                                          (side == 'values' and only_contains(simple_val, 'punct'))):
            continue
        # Note that this only passes on positive integers. Floats and negative integers will fall through!
        elif kwargs['rm_only_digits'] and ((side == 'keys' and simple_key.isdigit()) or
                                           (side == 'value' and simple_val.isdigit())):
            continue
        elif kwargs['rm_contains_nonan'] and ((side == 'keys' and contains(simple_key, 'nonan')) or
                                              (side == 'values' and contains(simple_val, 'nonan'))):
            continue
        elif kwargs['rm_contains_digits'] and ((side == 'keys' and contains(simple_key, 'digit')) or
                                               (side == 'values' and contains(simple_val, 'digit'))):
            continue
        elif kwargs['rm_contains_punct'] and ((side == 'keys' and contains(simple_key, 'punct')) or
                                              (side == 'values' and contains(simple_val, 'punct'))):
            continue

        clean_dict[key] = val

    return clean_dict


def sort_simple_dict(simple_dict, sort_on='keys', **kwargs):
    default_params = {'reverse': False}
    kwargs = verify_kwargs(default_params, kwargs)

    if sort_on == 'keys':
        sorted_dict = sorted(simple_dict.items(), key=itemgetter(0), reverse=kwargs['reverse'])
    elif sort_on == 'values':
        sorted_dict = sorted(simple_dict.items(), key=itemgetter(1), reverse=kwargs['reverse'])
    else:
        raise ValueError(f"Unexpected value {sort_on} for sort_on. Use keys or values")

    # returns a list of tuples
    # dicts are not sorted, unless using Python 3.6 or higher
    return sorted_dict


def verify_kwargs(defaults, kwargs):
    for name, default_val in defaults.items():
        if name in kwargs:
            param = kwargs[name]
            param_type = type(default_val)
            if not isinstance(param, param_type):
                raise ValueError(f"Unexpected value {param} for {name}. A(n) {param_type.__name__} value is expected")
        else:
            kwargs[name] = default_val

    return kwargs
