# Pictor documentation and report

## Introduction
Pictor is a tiny and simply "pictures interactions predictor".
The input of the program is a csv file containing a dataset with ~350000 samples.

Each sample includes values for about 50 variables (features) of one picture. One of those features is the 'interactions' feature, which implies the amount of user interactions with that picture.

Using the functions and capabilities of scikit-learn in combination with NumPy and SciPy for the Python programming language, this program tries to predict the amount of user interactions of a picture with those features.

## Design
Pictor loads the input dataset, transforms it and use it values to call  scikit-learn prediction functions. The steps necessaries to perform this actions are:

1. Load input csv file.
2. Transform inner values types into proper types (*float* or *str*).
3. Perform some feature engineering and see if all of them are necessaries.
4. Split the dataset into inputs and expected outputs (*interactions* corresponding values).
5. Inspect input dataset and choose a proper algorithm.
6. Make the necessaries transformations to use the dataset with the choosen algorithm.
7. Split both input dataset and output (interactions) arrays into training and testing sets.
8. Make predictions with selected algorithm.
9. Adjust algorithm parameters and make predictions again.
10. See if results were as expected; if not, adjust parameters again and run a new prediction.
11. When prediction results were as expected, present some information to the user (score of prediction, feature engineering results, etc).

With that steps in mind, Pictor software were designed as modules with functions, that can be used independently.

The program consists in a main loop that call each function in the order needed. All functions developed have their corresponding step in evaluation. 

Considering that one function in the program should do *one function*, helper additional functions were needed for each case. In the same way, error cases should be considered and catched before to make program crash.

The project includes a "tools" folder for functions that are called from the main loop.

## Development
### Feature engineering
Some kind of feature engineering was performed inspecting the data, doing dot plots of "feature N" vs "interactions", and trying to make a prediction with each feature independently and comparing errors and results.

Some of the features were discarded because they add processing complication and do not offers important information to the final fit.

The preprocessing of features is performed using DictVectorizer from sklearn. This class takes a list of dicts, and produces a numpy array with all of their features in a suitable format (float in this case). Classification and string values are changed for one new feature for each class available. For example if a feature named XXX have the values AAA, BBB and CCC, then three new features replace the older one: 'XXX=AAA', 'XXX=BBB' and 'XXX=CCC', with the value 1 in the case it corresponds.

### Algorithm
The chosen predictor function is a regressor, considering the output (amount of interactions) as a continuous output. In the ideal case, those are scalar values, but the term "continuous" implies some kind of "relationship" between two values: for example 100 and 101 are very similar values for interactions count.

With this in mind, and taking in account the high number of samples and features, different regression functions were used, tested and discarded:
* SVR: Discarded: do not converge.
* NuSVR: Discarded: do not converge.
* Logistic Regressor: Discarded: do not converge.
* Linear Regressor: Converge, but the results weren't successful. The maximum prediction score is about 45%.
* DecissionTree: Converge and obtain a much better score and prediction.
* RandomForest: After different tests, this is the final choose because it combines the precision of different decision trees.

By adding more estimators (n_estimator parameter) to the RandomForest, it was seeing that the final score varies in a lower percentage, but adds more time complexity to the total computing time, so the final RandomForest was a combination of simplicity and performance.

### Results presentation
The user needs to know how accurate the program can be predicting interactions results with the provided input data. For this purpose, the software at the ends shows the score (in floating point number type) of the precision. Taking in account that 1.0 is the highest value that this variable can take. Also two accuracy meters are included: MSE and median absolute error.

In addition, and taking advantage of the *feature_importance* that the RandomForest can provide, the software shows to the user a bar chart with the 10 most *important* features, with their values normalized (better value is 1 for the highest important feature). For this chart, the user should have installed matplotlib for Python.

## Conclusion
The higher score (accuracy) obtained was about **82%**. It is a high precision for this kind of problems. In this case this was obtained using the whole set of samples, an amount of 25 DecissionTrees regressors.

This solution presents a cons: it is dependant of the amount of samples used. The prediction accuracy fall of as the amount of samples is reduced. With this amount of samples, the total processing time can rise up to 30 minutes or more, depending the kind of CPU, RAM memory and hardware resources you have available.

