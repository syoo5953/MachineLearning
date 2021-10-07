# classification-algorithms-comp3308
Solution to the second assignment of COMP3308: Introduction to AI. This assignment required an implementation of the K-Nearest Neighbour (Euclidean distance) and Naive Bayes classification algorithms. Ten-fold stratification and cross validation code has been included for 1NN, 5NN and NB, which is not a marked component of this assessment.

## Usage 
`./MyClassifier.py training_data.csv testing_data.csv <algorithm>`

Algorithms available are `NB` (Naive Bayes) and `<k>NN` (K-Nearest Neighbour).
Please see `pima.csv` or `pima-CFS.csv` and `testData.csv` for example training and testing data, respectively.

## Requirements
* Python 3.6.*
* NumPy

## TODO
* Update README for 10-fold cross validation usage.
