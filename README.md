# pictor

[Instructions here](https://www.facebook.com/emmanuel.naah.37)

Pictor is a 'pictures interactions predictor':
Given an input dataset, with about 50 different data from ~350000 pictures from social networks, pictor tries to predict the interactions over any other picture, with that same info.

## Execution requeriments:
  - Python 2.7
  - scikit-learn
  - NumPy
  - SciPy

## Execution instructions
In order to execute pictor, just call this module as argument of the python interpreter:
```sh
$ python pictor.py <args>
```
Were args can be:
  - --csv="<file>": Specify the input csv file. If it is not specified, then is assumed that the file is named 'consolidated_features.csv' and is located in the same path.
  - --feat="<file>": Specify the input file with features to discard. The file should be a plain text file with one feature name by file. Lines beginning with '#' are discarded. An example file is in the root path.
  - --verbose="y|n": Show verbose output. Verbose output include time duration for each function in the program. If it is not specified, then is assumed 'y'es.

For example:
```sh
$ python pictor.py --csv="../consolidated_features.csv" --feat="features_unwanted.csv"
```

## Tags:
[machine-learning] [ml], [sklearn] [sk], [python] [py], [prediction] [pr], [pictures] [pi], [interactions] [in]


[ml]: https://github.com/search?utf8=%E2%9C%93&q=machine-learning
[sk]: https://github.com/search?utf8=%E2%9C%93&q=sklearn
[py]: https://github.com/search?utf8=%E2%9C%93&q=python
[pr]: https://github.com/search?utf8=%E2%9C%93&q=prediction
[pi]: https://github.com/search?utf8=%E2%9C%93&q=pictures
[in]: https://github.com/search?utf8=%E2%9C%93&q=interactions
