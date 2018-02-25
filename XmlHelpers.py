from xml.etree.ElementTree import iterparse

from .TypeHelpers import verify_kwargs


def scan_xml_and_execute(file, exec_func, restrict_to_nodes=None, verbose=0):
    if verbose not in range(0, 3):
        raise ValueError(f"Unexpected value {verbose} for verbose. 0, 1, or 2 expected")

    if restrict_to_nodes is not None and not isinstance(restrict_to_nodes, list):
        raise ValueError(f"Unexpected value {restrict_to_nodes} for restrict_to_nodes. None or a list expected")

    node_i = 0
    for _, elem in iterparse(file):
        if restrict_to_nodes is not None and elem.tag not in restrict_to_nodes:
            elem.clear()
            continue

        if verbose > 0:
            proc_str = "Processing"
            if verbose > 1:
                proc_str += f" file {file}"
            proc_str += f" node {node_i+1}"
            print(proc_str, end="\r", flush=True)

        exec_func(elem)
        node_i = node_i + 1
        elem.clear()

    return None


def get_attr_frequencies(file, nodes, attr, restrict_to_pos=None, **kwargs):
    default_params = {'normalize_capitalisation': False, 'pos': 'pos', 'include_pos': False, 'verbose': 0}
    kwargs = verify_kwargs(default_params, kwargs)

    if restrict_to_pos is not None and not isinstance(restrict_to_pos, list):
        raise ValueError(f"Unexpected value {restrict_to_pos} for restrict_to_pos. None or a list expected")

    freq_d = {}

    def increment_attr_count(elem):
        attrs = elem.attrib

        if attr not in attrs:
            return None

        # If restric_to_pos, ensure postag in attr list, and postag value in restric_to_pos list
        if restrict_to_pos is not None and (kwargs['pos'] not in attrs or attrs[kwargs['pos']] not in restrict_to_pos):
            return None

        # If required postag not in attr list, return
        if kwargs['include_pos'] and kwargs['pos'] not in attrs:
            return None

        attr_val = attrs[attr].lower() if kwargs['normalize_capitalisation'] else attrs[attr]

        if kwargs['include_pos']:
            attr_val = (attr_val, attrs[kwargs['pos']])

        if attr_val not in freq_d:
            freq_d[attr_val] = 1
        else:
            freq_d[attr_val] = freq_d[attr_val] + 1

        return None

    scan_xml_and_execute(file, lambda xml_elem: increment_attr_count(xml_elem), restrict_to_nodes=nodes,
                         verbose=kwargs['verbose'])

    # Returns dict with value=>count, or (value,pos)=>count, depending on include_pos parameter
    return freq_d
