import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# # Preprocessing data
def cleanTrainData():
    trainDataFile = pd.read_csv('TrainOnMe_orig.csv')
    # remove first columns that are not required
    trainDataFile = trainDataFile.drop(trainDataFile.columns[0], axis=1)
    # remove 'x12' column since it is always true
    trainDataFile = trainDataFile.drop(['x12'], axis=1)
    # do one hot encoding on 'x7' column
    trainDataFile = pd.get_dummies(trainDataFile, columns=['x7'])
    print(trainDataFile.head())

    # Label encode 'y' column to convert 'Barbie' inspired names to numerical values
    labelEnc = LabelEncoder()
    labelEncY = labelEnc.fit(trainDataFile['y'])
    trainDataFile['y'] = labelEncY.transform(trainDataFile['y'])

    trainDataFile.to_csv('cleanTrainSet.csv', index=False)


def cleanTestData():
    testDataFile = pd.read_csv('EvaluateOnMe.csv')
    # remove first columns that are not required
    testDataFile = testDataFile.drop(testDataFile.columns[0], axis=1)
    # remove 'x12' column since it is always true
    testDataFile = testDataFile.drop(['x12'], axis=1)
    # do one hot encoding on 'x7' column
    testDataFile = pd.get_dummies(testDataFile, columns=['x7'])
    print(testDataFile.head())

    testDataFile.to_csv('cleanTestSet.csv', index=False)


cleanTrainData()
cleanTestData()

trainFile = pd.read_csv('cleanTrainSet.csv')
testFile = pd.read_csv('cleanTestSet.csv')


def TrainAndTest(classifier):
    """
    Run classifier with cross-validation and compute mean accuracy using cross_val_score
    """
    cv = StratifiedKFold(n_splits=10)
    X = trainFile.drop(['y'], axis=1)

    classifier.fit(X, trainFile['y'])
    # Compute the accuracy scores for each fold
    accuracy_scores = cross_val_score(classifier, X, trainFile['y'], cv=cv, scoring='accuracy')

    # Return the mean accuracy
    return np.mean(accuracy_scores)


RFClassifier = RandomForestClassifier(n_estimators=400, random_state=1)
print(TrainAndTest(RFClassifier))

Prediction = RFClassifier.predict(testFile)

labelMap = {0: 'Allan', 1: 'Barbie', 2: 'Ken'}
Prediction = [labelMap.get(x, x) for x in Prediction]

with open('PredictedLabels.txt', 'w', encoding='utf-8') as file:
    for pred in Prediction:
        file.write("%s\n" % pred)
