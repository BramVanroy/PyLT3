============
Installation
============

Installing PyLT3 (pronounced as *pilot* or *pilot-ee*) is as easy as 1, 2, *pylt3*! Haha... Ha... Using pip:

.. code:: bash

  pip install pylt3

=====
Notes
=====
First of all, this is a very young package that is continuously in development. Everything is subject to change.
Like, literally... *everything*. At the moment, the package is more a way for me to understand building and maintaining
Python packages than that it aims to be a full-fledged tool for the Python community. With that, I am very eager to
learn, so `pull requests`_, and `suggestions and issues`_ are always welcome.

.. _pull requests: https://github.com/BramVanroy/PyLT3/pulls
.. _suggestions and issues: https://github.com/BramVanroy/PyLT3/issues

*************************
Note on :code:`exec_func`
*************************

In :code:`exec_func` parameters in all functions below, it is recommended to use `lambda expressions`_ as a value for
:code:`exec_func` when using a function with more than one argument. By default, the current file
(:code:`scan_dir_and_execute`) or the current line (:code:`scan_file_and_execute`) is passed to :code:`exec_func` but
if you want to specify more arguments for your own function, you should use lambdas with the respective items as first
argument (cf. some of the examples below).

.. _lambda expressions: https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions

*************************
Note on keyword arguments
*************************

To easily type-check arguments, some keyword arguments are soaked up into :code:`kwargs`, meaning that you cannot add them as
positional arguments, i.e. the keyword is required (e.g. `recursive=True` rather than :code:`True`). In the descriptive
sections below, packed :code:`kwargs` are preceded by `**` so that you know you cannot use these as positional
arguments. Note that this is *not* the case for all keyword arguments. A keyword is added to kwargs only when simple
type-checking based on the default value is possible, and when there are no specific requirements for a value.

For instance, when code expects an optional argument to only be :code:`True` or :code:`False` (e.g. :code:`recursive`), it will be added
to the :code:`kwargs` dictionary. Because the default value (e.g. :code:`False`) is a boolean, only boolean types will be accepted.
However, a parameter such as :code:`verbose` that can reflect many levels of verboseness, e.g. `0`, `1`, and `2`, will not be
added. The default value `0` would mean that type-checking simply wants an :code:`int`, but for this example we only want
valid values of zero to two (including). In other words, if we do more than just type-checking but also
*value-checking*, the argument is *not* added to :code:`kwargs`. Note that only type-checking and being added to :code:`kwargs`,
*does* occur if a value is not subject to restrictions it may later receive when it is passed onto another function.
For an example, cf. :code:`get_attr_frequencies` which assumes :code:`verbose` to be in :code:`kwargs`, so only type-checking occurs. The
value of verbose is then fed to :code:`scan_xml_and_execute` as a regular argument, outside :code:`kwargs`, because in this
function :code:`verbose` needs a value check as well (it has to be `in range(0, 3)`).

Another case where an optional argument will not be added to :code:`kwargs` is when multiple types are possible for an
argument. A typical example is when an argument can be either :code:`None`, or a :code:`str` or :code:`list`. In these cases, the
argument is not added to :code:`kwargs` either.

In short: if, in the *current* function, we only want to ensure an argument's type and this type can be deducted from
its default value, and there is only one allowed type, then the argument is added to :code:`kwargs`, and it cannot be used as
a positional argument. In all other cases it can.

Also see the `Python documentation on arguments`_.

.. _Python documentation on arguments: https://docs.python.org/3/glossary.html#term-parameter

=======
Modules
=======

********************
:code:`file_helpers`
********************

:code:`scan_dir_and_execute`
============================
.. code:: python

  scan_dir_and_execute(root, exec_func, exclude_dirs=None, verbose=0, recursive=True)


Traverses a directory recursively (by default) and executes a user-defined function for every file that is encountered.

* :code:`root`: path to the root directory that will be processed (required)
* :code:`exec_func`: the function that is executed for every file. As stated above, it is only logical to use a lambda
  expression here (required)
* :code:`exclude_dirs`: an iterable containing the directory names that you wish to exclude from the recursion
  (`default=None`)
* :code:`verbose`: an integer indicating the level of verboseness. `0` will not print anything; `1` will print a message when
  entering a new directory; `2` will also print when processing a new file (`default=0`)
* ** :code:`recursive`: a boolean indicating whether to recursively go through all directories and sub-directories or not
  (`default=True`)

Returns :code:`None`


:code:`scan_file_and_execute`
=============================
.. code:: python

  scan_file_and_execute(file, exec_func, verbose=0, encoding=locale.getpreferredencoding())


Reads a file and executes a user-defined function for each line with the line itself and line number as default
arguments.

* :code:`file`: path to the file that will be processed (required)
* :code:`exec_func`: the function that is executed for every line. As stated above, it is only logical to use a lambda
  expression here (required)
* :code:`verbose`: an integer indicating the level of verboseness. `0` will not print anything; `1` will print the line
  number of the line that's currently being processed (in-place, i.e. ending with `\r`); `2` will shown the current
  file followed by the line number (also in-place) (`default=0`)
* ** :code:`encoding`: the encoding used for opening the file, as it would be used in an `open()` call
  (`default=locale.getpreferredencoding()`)

Returns :code:`None`


:code:`concatenate_files`
=========================
.. code:: python

  concatenate_files(input_item, output_file, extension=None, remove_headers=0, verbose=0, retain_first_header=False,
                    recursive=True, encoding=locale.getpreferredencoding())


Takes a list of files and concatenates them, or concatenates all files - optionally filtered by extension - in a given
directory.

* :code:`input_item`: either a list of files to concatenate or a directory as a string whose file contents will be
  concatenated (required)
* :code:`output_file`: the resulting output file (required)
* :code:`extension`: the extension to filter the files in case :code:`input_item` is a string. Only files in that directory ending
  with :code:`extension` will be concatenated (`default=None`)
* :code:`remove_headers`: an integer indicating which first lines of all files need to be removed. Useful in case all files
  share the same header row. The integer represents how many lines to skip (`default=0`)
* :code:`verbose`: an integer indicating the level of verboseness. `0` will not print anything; `1` will print the line
  number of the linethat's currently being processed (in-place, i.e. ending with `\r`); `2` will shown the current file
  followed by the line number (also in-place) (`default=0`)
* ** :code:`retain_first_header`: a boolean indicating whether or not the header lines of the first file need to be retained.
  In other words, when :code:`remove_headers` is set to an integer larger than `0` and `retain_first_header==True` then the
  resulting file will have only one remaining header (`default=False`)
* ** :code:`recursive`: a boolean indicating whether to recursively go through all directories and sub-directories or not
  (`default=True`)
* ** :code:`encoding`: the encoding used for opening the file, as it would be used in an `open()` call
  (`default=locale.getpreferredencoding()`)

Returns :code:`str`: the path to the file that has just been created, i.e. :code:`output_file`


:code:`print_simple_dict`
=========================
.. code:: python

  print_simple_dict(simple_dict, output_file, encoding=locale.getpreferredencoding())


Given a one-level dictionary, this function will print it to an output file as key-value pairs, separated by tabs. It
is possible to sort the dictionary by keys or values, and reverse the order.

* :code:`simple_dict`: dictionary that needs printing (required)
* :code:`output_file`: the resulting output file (required)
* ** :code:`encoding`: the encoding used for opening the file, as it would be used in an `open()` call
  (`default=locale.getpreferredencoding()`)

Returns :code:`str`: the path to the file that has just been created, i.e. :code:`output_file`


*******************
:code:`xml_helpers`
*******************

:code:`scan_xml_and_execute`
============================
.. code:: python

  scan_xml_and_execute(file, exec_func, restrict_to_nodes=None, verbose=0)


TODO: add arguments and options


:code:`get_attr_frequencies`
============================
.. code:: python

  get_attr_frequencies(file, nodes, attr, normalize_capitalisation=False, restrict_to_pos=None, pos='pos',
                       include_pos=False, verbose=0)


TODO: add arguments and options


********************
:code:`type_helpers`
********************

:code:`clean_simple_dict`
=========================
.. code:: python

  clean_simple_dict(simple_dict, side='key', rm_only_punct=False, rm_contains_punct=False, rm_only_digits=False,
                    rm_contains_digits=False, rm_only_nonan=False, rm_contains_nonan=False)


TODO: add arguments and options


:code:`sort_simple_dict`
========================
.. code:: python

  sort_simple_dict(simple_dict, sort_on='keys', reverse=False)


* :code:`sort_on`: sort the resulting dictionary and sort on :code:`keys` or :code:`value` (only these values and :code:`None` are accepted)
  (`default=None`)
* :code:`reverse`: a boolean that determines whether a sorted dictionary will be reserved or not (`default=False`)


Returns :code:`list`:

:code:`verify_kwargs`
=====================
.. code:: python

  verify_kwargs(defaults, kwargs, allow_none=None)


Given a dictionary of default key-value pairs, and another dictionary with user-defined values, it is ensured that the
type of user-defined values is the same as the default value's type. The function returns a merged dictionary with
:code:`kwargs` taken precedence over :code:`defaults`.

* :code:`defaults`: a dictionary containing default keys and respective values
* :code:`kwargs`: a dictionary that contains the actual values that you want to set
* :code:`allow_none`: a list of parameter names that next to their default value type can also accept None (`default=None`)

Returns :code:`dict`: the result of merging two dictionaries together
