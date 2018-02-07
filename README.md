# Local installation
1. Download or clone package PyLT3 (pronounced _pilot_ or _pilot-ee_) to your project folder;
2. import the module that you need, e.g.

`from PyLT3 import FileHelpers`

# Usage

In all functions below, it is recommended to use 
[lambda expressions](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) as a value for `exec_func`
when using a function with more than one argument. By default, the current file (`scandir_and_execute`) or the current 
line (`scanfile_and_execute`) is passed to `exec_func` but if you want to specify more arguments for your own function, 
you should use lambdas with the respective items as first argument (cf. some of the examples below). 


## FileHelpers
### `scandir_and_execute`
```python
scandir_and_execute(root, exec_func, exclude_dirs=None, recursive=True, verbose=0)
```

Traverses a directory recursively (by default) and executes a user-defined function for every file that is encountered. 

#### Arguments and options
* `root`: path to the root directory that will be processed (required)
* `exec_func`: the function that is executed for every file. As stated above, it is only logical to use a lambda 
expression here (required)
* `exclude_dirs`: an iterable containing the directory names that you wish to exclude from the recursion 
(`default=None`)
* `recursive`: a boolean indicating whether to recursively go through all directories and sub-directories or not
(`default=True`)
* `verbose`: an integer indicating the level of verboseness. `0` will not print anything; `1` will print a message when 
entering a new directory; `2` will also print when processing a new file
(`default=0`)

#### Examples
**1.** If you want to execute a function only for files with a given extension, you could do something like this:

```python
from PyLT3 import FileHelpers
 
def parse_file(file, extension):
    if file.endswith(f".{extension}"):
        print(file)
 
FileHelpers.scandir_and_execute(r"C:\my\data", lambda file: parse_file(file, "xml"))
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

FileHelpers.scandir_and_execute(r"C:\brown\corpus", lambda file: create_ngrams(file, 3), verbose=2)

for n in gram_freqs:
    for ngram in gram_freqs[n]:
        print(str(n) + " GRAM: " + str(ngram) + " - freq: " + str(gram_freqs[n][ngram]))
```


### `scanfile_and_execute`
```python
scanfile_and_execute(file, exec_func, encoding=locale.getpreferredencoding(), verbose=0)
```

Reads a file and executes a user-defined function for each line that is read.

#### Arguments and options
* `file`: path to the file that will be processed (required)
* `exec_func`: the function that is executed for every line. As stated above, it is only logical to use a lambda 
expression here (required)
* `encoding`: the encoding used for opening the file, as it would be used in an `open()` call
(`default=locale.getpreferredencoding()`, which is the default for file IO)
* `verbose`: an integer indicating the level of verboseness. `0` will not print anything; `1` will print the line number
of the line that's currently being processed (in-place, i.e. ending with `\r`); `2` will shown the current file 
followed by the line number (also in-place)
(`default=0`)

#### Examples
**1.** If you want to replace certain occurrences in a file, for instance because you want to make collocations by 
delimiting a keyword by tabs, you can easily do that with this function. In the output file, the keyword will be 
surrounded by tabs so that tools such as AntConc can easily recognise it.


```python
from PyLT3 import FileHelpers
import re

def collocate_out(line, word):
    replaced = re.sub(r"\b(%ss?)\b" % re.escape(word), r"\t\1\t", line, flags=re.IGNORECASE)
    OUT.write(replaced)

OUT = open("collocation.txt", "w")
FileHelpers.scanfile_and_execute(r"C:\my\cookies.txt", lambda line: collocate_out(line, "cookie"))
OUT.close()
```


### `scanfile_and_execute`
```python
concatenate_files(input_item, output_file, extension=False, remove_headers=False, retain_first_header=False,
    recursive=True, verbose=0)
```

Takes a list of files and concatenates them, or concatenates all files - optionally filtered by extension - in a given 
directory. 