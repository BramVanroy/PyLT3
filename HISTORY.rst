#######
History
#######

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
* Added optional parameter :code:`allow_none` to :code:`verify_kwargs` in :code:`type_helpers`. This allows a user to give a
  list of names of the keyword arguments that can be None in addition to their default type.
* TODO: complete and add examples
