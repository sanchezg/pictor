# Pictor documentation and report

## Introduction
Pictor is a tiny and simply "pictures interactions predictor".
The input of the program is a csv file containing a dataset with >340000 samples.

Each sample includes values for about 50 variables (features) of one picture. One of those features is the 'interactions' feature, which implies the amount of user interactions with that picture.

Using the functions and capabilities of scikit-learn in combination with NumPy and SciPy for the Python programming language, this program tries to predict the amount of user interactions of a picture with those features.

## Project structure

pictor/
    | data/     Contains input data used by the modules.
    | out/      Contains output data produced by the execution of the software included.
    | src/      Contains main project and related modules. The additional dataset_analysis.py includes an extra program to analyse the data.

## Design
Pictor loads the input dataset, transforms it and use it values to call  scikit-learn prediction functions. The steps necessaries to perform this actions are:

1. Load input csv file.
2. Transform inner values types into proper types (*float* or *str*).
3. Perform some feature engineering.
4. Split the dataset into inputs and expected outputs (*interactions* corresponding values).
5. Inspect input dataset and choose a proper algorithm.
6. Make the necessaries transformations to use the dataset with the choosen algorithm.
7. Split both input dataset and output (interactions) arrays into training and testing sets.
8. Make predictions with selected algorithm.
9. Adjust algorithm parameters and make predictions again.
10. See if results were as expected; if not, adjust parameters again and run a new prediction.
11. When prediction results were as expected, present some information to the user (score of prediction, feature engineering results, etc).

With that steps in mind, Pictor software was designed with two main classes (from the concept of class in OOP): DatasetExplorer and Predictor.

The program consists in a main flow that creates a DatasetExplorer and a Predictor objects. Then the main flow calls the methods of DatasetExplorer in order to perform feature engineering: analyse samples and features, fills empty values in some samples, discard samples or features as they aren't important, or join some features values in one (categorize or recategorize features). All functions developed have their corresponding step in evaluation. 

Considering that one function in the program should do *one function*, helper additional functions were needed in some cases. For this purpose, an additional module called helper_functions.py is used.

An extra module called "dataset_analysis.py" is attached. This module is not used by the main flow, but this module was used to perform some analysis, and can be read to understand some decisions taken in the main flow of the project. Using this module you can obtain the histograms and bar charts located in out/ folder.

## Development
### Feature engineering
Feature engineering involved imply to: 
* Analyse continuous features and obtain mean, median, 1st and 3rd percentile, std deviation, cardinality, missing, max and min values. Use of histograms.
* Analyse categorical features and obtain 1st and 2nd modes and their frequencies, cardinality and missing values. Use of bar plots.
* Analyse of feature correlation: use of pearson coefficient and scatter plots.
* Dataset inspection and data quality issues detection and handling.
* Inspection for outliers: what cases are valid outliers and what cases not.
* Outliers replace by thresholds.
* Samples with outliers remove.
* Fill of missing values in categorical data with 1st mode.
* Fill of missing values in continuous data with mean.
* Analyse with model and identify "strong" importances features and "weak" importances.
* Join or replacement of "weak" importances features values.

Most of those actions were performed more than one time and in different combination each one with others.

Some of the features were discarded because they add processing complication and do not offers important information to the final fit.

Then, the preprocessing of features is performed using DictVectorizer from sklearn. This class takes a list of dicts, and produces a numpy array with all of their features in a suitable format (float in this case). Classification and string values are changed for one new feature for each class available. For example if a feature named XXX have the values AAA, BBB and CCC, then three new features replace the older one: 'XXX=AAA', 'XXX=BBB' and 'XXX=CCC', with the value 1 in the case it corresponds.

Resulting histograms and bar charts are located in data/ folder. Analysis results are attached in a spreadsheet outside the project. You can access this spreadsheet at:
https://docs.google.com/spreadsheets/d/1dmdpQkNEuK300s1GJ8VTUA2kKcYWfyC3GNVvorJEXng/edit?usp=sharing

### Algorithm
The chosen predictor function is a regressor, considering the output (amount of interactions) as a continuous output. In the ideal case, those are scalar values, but the term "continuous" implies some kind of "relationship" between two values: for example 100 and 101 are very similar values for interactions count.

With this in mind, and taking in account the high number of samples and features, different regression functions were used, tested and discarded:
* SVR: Discarded: do not converge.
* NuSVR: Discarded: do not converge.
* Logistic Regressor: Discarded: do not converge.
* Linear Regressor: Converge, but the results weren't successful. The maximum prediction score is about 45%.
* DecissionTree: Converge and obtain a much better score and prediction.
* RandomForest: This is a very good choice considering that it can be run in parallel and a good fit can be obtained with very low work. But this predictor tends to overfit the model.
* GradientBoostingTree: After different tests this was the final choice because it avoids overfitting the data.

By adding more estimators (n_estimator parameter) to the GradientBoosting a better score can be obtained but the processing time is increased. So a final estimators number was 100.

### Results presentation
The user needs to know how accurate the program can be predicting pictures interactions results with the provided input data. For this purpose, the software at the ends shows the score (in floating point number type) of the precision. This score represents the percentage of predicted values, and is calculated executing the model in a different set that the used for building the model.

For evaluate how good the model is, the metric used is MSE (mean squared error). This value is calculated both in the data used to build the model and the data used to test the model.

### Results obtained
After running the main program, the software shows the results to the user using the console out:

Modeling predictor... 
DONE. Time modeling predictor: 1693.05s.
MSE on train set: 192666.147
Testing predictor... 
DONE. Time testing predictor: 1.34s.
Predictor score: 0.715
MSE on test set: 211704.575

## Conclusion
The higher score (accuracy) obtained was about **71.5%**. It is a high precision for this kind of problems. In this case it was obtained using the whole set of samples, and the following parameters for the predictor:

'max depth' = 3
'n_estimators' = 100

Obtaining similar values when calculating MSE (at least values in the same order of magnitude) indicates that the model is not overfitted, and works in a similar way (pretty well) on both set of data: data used to build the model (fit the algorithm) and data used to evaluate the model (predict results and calculate score).
