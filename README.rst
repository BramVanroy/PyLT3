**!! README IS NOT UP-TO-DATE !!**

# Contents

- [Local installation](#local-installation)
- [Usage](#usage)
  * [FileHelpers](#filehelpers)
    + [`scan_dir_and_execute`](#scan-dir-and-execute)
      - [Arguments and options](#arguments-and-options)
      - [Examples](#examples)
    + [`scan_file_and_execute`](#scan-file-and-execute)
      - [Arguments and options](#arguments-and-options-1)
      - [Examples](#examples-1)
    + [`concatenate_files`](#concatenate-files)
      - [Arguments and options](#arguments-and-options-2)
      - [Examples](#examples-2)
    + [`print_simple_dict`](#print-simple-dict)
      - [Arguments and options](#arguments-and-options-3)
      - [Examples](#examples-3)
  * [XmlHelpers](#xmlhelpers)
    + [`scan_xml_and_execute`](#scan-xml-and-execute)
      - [Arguments and options](#arguments-and-options-4)
      - [Examples](#examples-4)
    + [`get_attr_frequencies`](#get-attr-frequencies)
      - [Arguments and options](#arguments-and-options-5)
      - [Examples](#examples-5)
  * [TypeHelpers](#typehelpers)
    + [`clean_simple_dict`](#clean-simple-dict)
      - [Arguments and options](#arguments-and-options-6)
      - [Examples](#examples-6)

# Local installation
1. Download or clone package PyLT3 (pronounced *pilot* or *pilot-ee*) to your project folder;
2. import the module that you need, e.g.

`from PyLT3 import FileHelpers`

# Usage
## Notes
**Note on `exec_func`**

In `exec_func` parameters in all functions below, it is recommended to use
[lambda expressions](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) as a value for `exec_func`
when using a function with more than one argument. By default, the current file (`scan_dir_and_execute`) or the current
line (`scan_file_and_execute`) is passed to `exec_func` but if you want to specify more arguments for your own function,
you should use lambdas with the respective items as first argument (cf. some of the examples below).

**Note on keyword arguments**

To easily type-check arguments, some keyword arguments are soaked up into `kwargs`, meaning that you cannot add them as
positional arguments, i.e. the keyword is required (e.g. `recursive=True` rather than `True`). In the *Arguments and
options* sections below, packed `kwargs` are preceded by `**` so that you know you cannot use these as positional
arguments. Note that this is *not* the case for all keyword arguments. A keyword is added to kwargs only when simple
type-checking based on the default value is possible, and when there are no specific requirements for a value.

For instance, when code expects an optional argument to only be `True` or `False` (e.g. `recursive`), it will be added
to the `kwargs` dictionary. Because the default value (e.g. `False`) is a boolean, only boolean types will be accepted.
However, a parameter such as `verbose` that can reflect many levels of verboseness, e.g. `0`, `1`, and `2`, will not be
added. The default value `0` would mean that type-checki