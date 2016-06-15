# pictor

Pictor is a 'pictures interactions predictor':
Given an input dataset, with about 50 different data from ~350000 pictures from social networks, pictor tries to predict the interactions over any other picture, with that same info.

## Execution requeriments:
A requirements.txt file is included. This file contains the python pip dependencies. In order to execute the script, you need to have installed virtualenv. 

Then create a new virtualenv by executing:
```sh
$ virtualenv some_new_venv
```

Activate the virtualenv just created:
```sh
$ source some_new_venv/bin/activate
```

Clone this project into this virtualenv, and install requirements by executing:
```sh
$ pip install -r requirements.txt
```

This will install the following dependencies inside locally to the virtualenv:
  - scikit-learn
  - NumPy
  - SciPy
  - matplotlib

## Execution instructions
In order to execute pictor, just call this module as argument of the python interpreter:
```sh
$ python pictor.py <args>
```
Were args must be:
  - --csv="<file>": Specify the input csv file. If it is not specified, then is assumed that the file is named 'consolidated_features.csv' and is located in the same path.
  - --feat="<file>": Specify the input file with features to discard. The file should be a plain text file with one feature name by file. Lines beginning with '#' are discarded. The file used with this project is 'features_to_discard.txt' and is located inside data folder.

For example:
```sh
$ python pictor.py --csv="../consolidated_features.csv" --feat="../data/features_to_discard.txt"
```

## Tags:
[machine-learning] [ml], [sklearn] [sk], [python] [py], [prediction] [pr], [pictures] [pi], [interactions] [in]


[ml]: https://github.com/search?utf8=%E2%9C%93&q=machine-learning
[sk]: https://github.com/search?utf8=%E2%9C%93&q=sklearn
[py]: https://github.com/search?utf8=%E2%9C%93&q=python
[pr]: https://github.com/search?utf8=%E2%9C%93&q=prediction
[pi]: https://github.com/search?utf8=%E2%9C%93&q=pictures
[in]: https://github.com/search?utf8=%E2%9C%93&q=interactions
