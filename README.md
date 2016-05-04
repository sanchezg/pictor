# pictor

This is the main file for the pictor project.

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
$ python pictor.py
```

If no additional arguments are provided, pictor will lookup in the current folder the 'consolidated_features.csv' file with all the data.
This file can be passed as parameter with the --csv flag:
```sh
$ python pictor.py --csv='../consolidated_features.csv'
```

## Tags:
[machine-learning] [ml], [sklearn] [sk], [python] [py], [prediction] [pr], [pictures] [pi], [interactions] [in]


[ml]: https://github.com/search?utf8=%E2%9C%93&q=machine-learning
[sk]: https://github.com/search?utf8=%E2%9C%93&q=sklearn
[py]: https://github.com/search?utf8=%E2%9C%93&q=python
[pr]: https://github.com/search?utf8=%E2%9C%93&q=prediction
[pi]: https://github.com/search?utf8=%E2%9C%93&q=pictures
[in]: https://github.com/search?utf8=%E2%9C%93&q=interactions
