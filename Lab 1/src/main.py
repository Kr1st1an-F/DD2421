import random

import monkdata as m
import dtree as d
import numpy as np
import matplotlib.pyplot as plt


def average(lst):
    return sum(lst) / len(lst)


def assignment1():
    print("Entropy Monk1: ", d.entropy(m.monk1))
    print("Entropy Monk2: ", d.entropy(m.monk2))
    print("Entropy Monk3: ", d.entropy(m.monk3))


def assignment3():
    datasets = [m.monk1, m.monk2, m.monk3]
    for dataset in datasets:
        print("Expected Information Gain")
        for i in range(6):
            print("\t", d.averageGain(dataset, m.attributes[i]))


def assignment5():
    print("Monk1")
    tM1 = d.buildTree(m.monk1, m.attributes)
    print(d.check(tM1, m.monk1))
    print(d.check(tM1, m.monk1test))
    print("Monk2")
    tM2 = d.buildTree(m.monk2, m.attributes)
    print(d.check(tM2, m.monk2))
    print(d.check(tM2, m.monk2test))
    print("Monk3")
    tM3 = d.buildTree(m.monk3, m.attributes)
    print(d.check(tM3, m.monk3))
    print(d.check(tM3, m.monk3test))


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def assignment7():
    datasets = [m.monk1, m.monk3]
    meanPerFraction = []
    mean = []
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    iterations = 100

    for dataset in datasets:
        for fraction in fractions:
            for n in range(iterations):
                if dataset == m.monk1:
                    meanPerFraction.append(d.check(prunedTree(dataset, fraction), m.monk1test))
                else:
                    meanPerFraction.append(d.check(prunedTree(dataset, fraction), m.monk3test))

            mean.append(average(meanPerFraction))
            meanPerFraction.clear()

    plotAssignment7(mean, fractions)


def chooseBestTree(trees, valSet):
    maxAccuracy = 0.0
    index = -1
    for i in range(0, len(trees)):
        currAccuracy = d.check(trees[i], valSet)
        if currAccuracy > maxAccuracy:
            maxAccuracy = currAccuracy
            index = i
    return trees[index]


def prunedTree(dataset, fraction):
    trainSet, valSet = partition(dataset, fraction)
    tree = d.buildTree(trainSet, m.attributes)
    validationPerformance = d.check(tree, valSet)
    newValidationPerformance = 1.1
    bestTree = tree
    while validationPerformance < newValidationPerformance:
        tree = bestTree
        validationPerformance = d.check(tree, valSet)
        newTrees = d.allPruned(tree)
        bestTree = chooseBestTree(newTrees, valSet)
        newValidationPerformance = d.check(bestTree, valSet)
    return tree


def plotAssignment7(mean, fractions):
    m1Mean = mean[:6]  # split mean, one for monk 1 other for monk 3
    m3Mean = mean[6:]

    plt.figure()
    plt.title("Pruning Effect on the Test Error for MONK-1 and MONK-3 datasets")
    plt.xlabel("Fraction of training and validation set")
    plt.ylabel("Accuracy")
    plt.plot(fractions, m1Mean, marker="o", label="MONK-1")
    plt.plot(fractions, m3Mean, marker="o", label="MONK-3")
    plt.legend()

    plt.show()


def main():
    # assignment1()
    # assignment3()
    # assignment5()
    assignment7()


if __name__ == "__main__":
    main()
