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
added. The default value `0` would mean that type-checking simply wants an `int`, but for this example we only want 
valid values of zero to two (including). In other words, if we do more than just type-checking but also 
*value-checking*, the argument is *not* added to `kwargs`. Note that only type-checking and being added to `kwargs`, 
*does* occur if a value is not subject to restrictions it may later receive when it is passed onto another function. 
For an example, cf. [`get_attr_frequencies`](#get-attr-frequencies) which assumes `verbose` to be in `kwargs`, so only 
type-checking occurs. The value of verbose is then fed to `scan_xml_and_execute` as a regular argument, outside 
`kwargs`, because in this function `verbose` needs a value check as well (it has to be `in range(0, 3)`).

Another case where an optional argument will not be added to `kwargs` is when multiple types are possible for an 
argument. A typical example is when an argument can be either `None`, or a `str` or `list`. In these cases, the 
argument is not added to `kwargs` either.

In short: if, in the *current* function, we only want to ensure an argument's type and this type can be deducted from 
its default value, and there is only one allowed type, then the argument is added to `kwargs`, and it cannot be used as 
a positional argument. In all other cases it can.

Also see the Python [documentation on arguments](https://docs.python.org/3/glossary.html#term-parameter).


## FileHelpers
### `scan_dir_and_execute`
```python
scan_dir_and_execute(root, exec_func, exclude_dirs=None, verbose=0, recursive=True)
```

Traverses a directory recursively (by default) and executes a user-defined function for every file that is encountered. 

#### Arguments and options
* `root`: path to the root directory that will be processed (required)
* `exec_func`: the function that is executed for every file. As stated above, it is only logical to use a lambda 
expression here (required)
* `exclude_dirs`: an iterable containing the directory names that you wish to exclude from the recursion 
(`default=None`)
* `verbose`: an integer indicating the level of verboseness. `0` will not print anything; `1` will print a message when 
entering a new directory; `2` will also print when processing a new file
(`default=0`)
* **`recursive`: a boolean indicating whether to recursively go through all directories and sub-directories or not
(`default=True`)

Returns `None`

#### Examples
**1.** If you want to execute a function only for files with a given extension, you could do something like this:

```python
from PyLT3 import FileHelpers
 
def parse_file(file, extension):
    if file.endswith(f".{extension}"):
        print(file)
 
FileHelpers.scan_dir_and_execute(r"C:\my\data", lambda file: parse_file(file, "xml"))
``` 


**2.** Given a directory and an integer, get a dictionary with all n-grams up to the given integer. For instance, if you
want unigrams, bigrams, and trigrams then the integer should be `3`. The resulting dictionary has as primary keys these 
integers (e.g. `1`, `2`, and `3`) and as value a Counter object - but it can be iterated as if it is an embedded 
dictionary.


```python
from PyLT3 import FileHelpers
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

gram_freqs = {}

def create_ngrams(file, i):
    global gram_freqs

    for n in range(1, i+1):
        if n not in gram_freqs:
            gram_freqs[n] = Counter([])

        text = open(file).read()
        tokenised = word_tokenize(text)
        grams = ngrams(tokenised, n)
        gram_freqs[n] += Counter(grams)

FileHelpers.scan_dir_and_execute(r"C:\brown\corpus", lambda file: create_ngrams(file, 3), verbose=2)

for n in gram_freqs:
    for ngram in gram_freqs[n]:
        print(str(n) + " GRAM: " + str(ngram) + " - freq: " + str(gram_freqs[n][ngram]))
```


### `scan_file_and_execute`
```python
scan_file_and_execute(file, exec_func, verbose=0, encoding=locale.getpreferredencoding())
```

Reads a file and executes a user-defined function for each line with the line itself and line number as default
arguments.

#### Arguments and options
* `file`: path to the file that will be processed (required)
* `exec_func`: the function that is executed for every line. As stated above, it is only logical to use a lambda 
expression here (required)
* `verbose`: an integer indicating the level of verboseness. `0` will not print anything; `1` will print the line number
of the line that's currently being processed (in-place, i.e. ending with `\r`); `2` will shown the current file 
followed by the line number (also in-place)
(`default=0`)
* **`encoding`: the encoding used for opening the file, as it would be used in an `open()` call
(`default=locale.getpreferredencoding()`

Returns `None`

#### Examples
**1.** If you want to replace certain occurrences in a file, for instance because you want to make collocations by 
delimiting a keyword by tabs, you can easily do that with this function. In the output file, the keyword will be 
surrounded by tabs so that tools such as AntConc can easily recognise it.


```python
from PyLT3 import FileHelpers
import re

def collocate_out(line, index, word):
    # Skip header line
    if index > 0:
        replaced = re.sub(r"\b(%ss?)\b" % re.escape(word), r"\t\1\t", line, flags=re.IGNORECASE)
        OUT.write(replaced)

OUT = open("collocation.txt", "w")
FileHelpers.scan_file_and_execute(r"C:\my\cookies.txt", lambda line, index: collocate_out(line, index, "cookie"))
OUT.close()
```


### `concatenate_files`
```python
concatenate_files(input_item, output_file, extension=None, remove_headers=0, verbose=0, retain_first_header=False,
                  recursive=True, encoding=locale.getpreferredencoding())
```

Takes a list of files and concatenates them, or concatenates all files - optionally filtered by extension - in a given 
directory.

#### Arguments and options
* `input_item`: either a list of files to concatenate or a directory as a string whose file contents will be 
concatenated (required)
* `output_file`: the resulting output file (required)
* `extension`: the extension to filter the files in case `input_item` is a string. Only files in that directory ending 
with `extension` will be concatenated (`default=None`)
* `remove_headers`: an integer indicating which first lines of all files need to be removed. Useful in case all files 
share the same header row. The integer represents how many lines to skip (`default=0`)
* `verbose`: an integer indicating the level of verboseness. `0` will not print anything; `1` will print the line number
of the line that's currently being processed (in-place, i.e. ending with `\r`); `2` will shown the current file 
followed by the line number (also in-place)
(`default=0`)
* **`retain_first_header`: a boolean indicating whether or not the header lines of the first file need to be retained. 
In other words, when `remove_headers` is set to an integer larger than `0` and `retain_first_header==True` then the 
resulting file will have only one remaining header (`default=False`)
* **`recursive`: a boolean indicating whether to recursively go through all directories and sub-directories or not
(`default=True`)
* **`encoding`: the encoding used for opening the file, as it would be used in an `open()` call
(`default=locale.getpreferredencoding()`

Returns `output_file`: the path to the file that has just been created

#### Examples

TODO: add examples

### `print_simple_dict`
```python
print_simple_dict(simple_dict, output_file, encoding=locale.getpreferredencoding())
```

Given a one-level dictionary, this function will print it to an output file as key-value pairs, separated by tabs. It 
is possible to sort the dictionary by keys or values, and reverse the order.

#### Arguments and options
* `simple_dict`: dictionary that needs printing (required)
* `output_file`: the resulting output file (required)
* **`encoding`: the encoding used for opening the file, as it would be used in an `open()` call
(`default=locale.getpreferredencoding()`, which is the default for file IO)

Returns `output_file`: the path to the file that has just been created

#### Examples
TODO: add examples

## XmlHelpers
### `scan_xml_and_execute`
```python
scan_xml_and_execute(file, exec_func, restrict_to_nodes=None, verbose=0)
```

#### Arguments and options
TODO: add arguments and options

#### Examples
TODO: add examples

### `get_attr_frequencies`
```python
get_attr_frequencies(file, nodes, attr, normalize_capitalisation=False, restrict_to_pos=None, pos='pos', verbose=0)
```

#### Arguments and options
TODO: add arguments and options

#### Examples
TODO: add examples

## TypeHelpers
### `clean_simple_dict`
```python
clean_simple_dict(simple_dict, side='key', rm_only_punct=False, rm_contains_punct=False, rm_only_digits=False,
                  rm_contains_digits=False, rm_only_nonan=False, rm_contains_nonan=False)
```

#### Arguments and options
TODO: add arguments and options

#### Examples
TODO: add examples

### `sort_simple_dict`

#### Arguments and options
* `sort_on`: sort the resulting dictionary and sort on `keys` or `value` (only these values and `None` are accepted)
(`default=None`)
* `reverse`: a boolean that determines whether a sorted dictionary will be reserved or not (`default=False`)

#### Examples
TODO: add examples
