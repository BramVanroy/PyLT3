import string


def clean_simple_dict(simple_dict, side='key', rm_only_punct=False, rm_contains_punct=False, rm_only_digits=False,
                      rm_contains_digits=False, rm_only_nonan=False, rm_contains_nonan=False):
    if not (side == 'key' or side == 'val'):
        raise ValueError(f'Unexpected value {side} for side. Use key or val')

    for param in [rm_only_punct, rm_contains_punct, rm_contains_punct, rm_only_digits, rm_contains_digits,
                  rm_only_nonan, rm_contains_nonan]:
        if not isinstance(param, bool):
            raise ValueError(f'Unexpected value {param} in params. A boolean value is required')

    def contains(token, contain_type):
        for char in token:
            if contain_type == 'nonan' and not char.isalnum():
                return True
            elif contain_type == 'digit' and char.isdigit():
                return True
            elif contain_type == 'punct' and char in string.punctuation:
                return True

        return False

    def only_contains(token, contain_type):
        for char in token:
            if contain_type == 'nonan' and char.isalnum():
                return False
            elif contain_type == 'punct' and char not in string.punctuation:
                return False

        return True

    clean_dict = {}

    for key, val in simple_dict.items():
        if rm_only_nonan and ((side == 'key' and only_contains(key, 'nonan')) or
                              (side == 'val' and only_contains(val, 'nonan'))):
            continue
        elif rm_only_punct and ((side == 'key' and only_contains(key, 'punct')) or
                                (side == 'val' and only_contains(val, 'punct'))):
            continue
        # Note that this only passes on positive integers. Floats and negative integers will fall through!
        elif rm_only_digits and ((side == 'key' and key.isdigit()) or (side == 'val' and val.isdigit())):
            continue
        elif rm_contains_nonan and ((side == 'key' and contains(key, 'nonan')) or
                                    (side == 'val' and contains(val, 'nonan'))):
            continue
        elif rm_contains_digits and ((side == 'key' and contains(key, 'digit')) or
                                     (side == 'val' and contains(val, 'digit'))):
            continue
        elif rm_contains_punct and ((side == 'key' and contains(key, 'punct')) or
                                    (side == 'val' and contains(val, 'punct'))):
            continue

        clean_dict[key] = val

    return clean_dict
