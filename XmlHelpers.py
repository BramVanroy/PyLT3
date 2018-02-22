import xml.etree.ElementTree as Et


def scan_xml_and_execute(file, exec_func, restrict_to_nodes=None, verbose=0):
    if verbose not in range(0, 3):
        raise ValueError(f'Unexpected value {verbose} for verbose')

    if restrict_to_nodes is not None:
        if isinstance(restrict_to_nodes, str):
            raise ValueError(f'Unexpected value {restrict_to_nodes} for restrict_to_nodes. None or a list expected')

    node_i = 0
    for _, elem in Et.iterparse(file):
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


def get_attr_frequencies(file, nodes, attr, normalize_capitalisation=False, restrict_to_pos=None, pos='pos', verbose=0):
    if restrict_to_pos is not None and isinstance(restrict_to_pos, str):
        raise ValueError(f'Unexpected value {restrict_to_pos} for restrict_to_pos. None or a list expected')

    freq_d = {}

    def increment_attr_count(elem):
        attrs = elem.attrib
        if attr not in attrs:
            return None

        if restrict_to_pos is not None and (pos not in attrs or attrs[pos] not in restrict_to_pos):
            return None

        attr_val = attrs[attr].lower() if normalize_capitalisation else attrs[attr]

        if attr_val not in freq_d:
            freq_d[attr_val] = 1
        else:
            freq_d[attr_val] = freq_d[attr_val] + 1

        return None

    scan_xml_and_execute(file, lambda xml_elem: increment_attr_count(xml_elem), restrict_to_nodes=nodes, verbose=verbose)

    return freq_d


