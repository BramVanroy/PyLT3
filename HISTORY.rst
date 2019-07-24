#######
History
#######

**************************
0.7.4 (July 24th, 2019)
**************************
* Removed :code:`Path().path` in favor of :code:`str(Path())`

**************************
0.7.3 (January 18th, 2019)
**************************
* Added :code:`normalize_digits` to :code:`preprocessing`. This tiny function replaces all digits by ones in a file.

*************************
0.7.1 (October 2nd, 2018)
*************************
* Changed the :code:`extension` parameter for :code:`concatenate_files` so that the dot has to be provided as well.
  This allows you to basically look for filenames ending with any string rather than just a final extension.

****************************
0.7.0 (September 27th, 2018)
****************************
* Added the :code:`naive` tokenizer option when tokenizing in :code:`preprocessing.py`. This ensures that external
  libraries need not be downloaded. The naive tokenizer does two things: replace word boundaries by a single
  whitespace character, and substitute sequences of whitespaces with a single whitespace.

****************************
0.6.6 (September 25th, 2018)
****************************
* BUGFIX in :code:`print_simple_dict()` inside :code:`file_helpers`.

****************************
0.6.5 (September 22nd, 2018)
****************************
* Added the :code:`keep_empty` option to :code:`tokenize_file` in :code:`preprocessing.py` so that empty lines are
  still printed when required (default: :code:`True`).

****************************
0.6.0 (September 21th, 2018)
****************************
* Renamed :code:`tokenize` to :code:`tokenize_file` in :code:`preprocessing.py`.
* Added :code:`tokenize_string` in :code:`preprocessing.py`.

****************************
0.5.0 (September 20th, 2018)
****************************
* Added :code:`preprocessing.py`. This still needs documentation, though.

***********************
0.4.0 (April 4th, 2018)
***********************
* Added optional parameter :code:`events` to :code:`scan_xml_and_execute` in :code:`xml_helpers`. This gives the user
  access to the events of iterparse


***********************
0.3.0 (March 2nd, 2018)
***********************
* First real release on PyPI.
* Added optional parameter :code:`allow_none` to :code:`verify_kwargs` in :code:`type_helpers`. This allows a user to
  give a list of names of the keyword arguments that can be None in addition to their default type.
* TODO: complete and add examples
