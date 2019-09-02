from string import punctuation


def is_simple_list(li):
    return isinstance(li, list) or isinstance(li, tuple)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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


def sort_iterable(iterable, sort_on=None, **kwargs):
    default_params = {'reverse': False, 'num_as_num': False}
    kwargs = verify_kwargs(default_params, kwargs)

    def _sort(i):
        if kwargs['num_as_num']:
            try:
                if i is None:
                    return sorted(iterable, key=float, reverse=kwargs['reverse'])
                else:
                    return dict(sorted(iterable.items(), key=lambda v: float(v[i]), reverse=kwargs['reverse']))
            except TypeError:
                raise TypeError(f"Tried parsing as float but could not. Only use num_as_num when all items that need "
                                f"to be sorted can be converted to float")
        else:
            if i is None:
                return sorted(iterable, key=str, reverse=kwargs['reverse'])
            else:
                return dict(sorted(iterable.items(), key=lambda v: str(v[i]), reverse=kwargs['reverse']))

    if isinstance(iterable, list):
        return _sort(None)
    elif isinstance(iterable, tuple):
        return tuple(_sort(None))
    elif isinstance(iterable, dict):
        if sort_on.lower() == 'keys':
            return _sort(0)
        elif sort_on.lower() == 'values':
            return _sort(1)
        else:
            raise ValueError(f"Unexpected value {sort_on} for sort_on. When sorting a dict, use keys or values")
    else:
        raise TypeError(f"Unexpected type {type(iterable)} for iterable. Expected a list, tuple, or dict")


def verify_kwargs(defaults, kwargs, allow_none=None):
    if allow_none is None:
        allow_none = []
    elif not isinstance(allow_none, list):
        raise ValueError("Expected value None or a list for allow_none")

    for name, default_val in defaults.items():
        if name in kwargs:
            param = kwargs[name]
            param_type = type(default_val)
            if not (isinstance(param, param_type) or name in allow_none and param is None):
                none_str = 'or None ' if name in allow_none else ''
                print(param)
                raise ValueError(f"Unexpected value {param} for {name}. A(n) {param_type.__name__} value "
                                 f"{none_str}is expected")
        else:
            kwargs[name] = default_val

    return kwargs
