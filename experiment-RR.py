#!/usr/bin/env python3

from data import adult, german, singles
from relabeling import randomOneSideRelabelData
import boosting
import svm
import lr
from errorfunctions import signedStatisticalParity, labelError, individualFairness
from utils import arrayErrorBars, errorBars, experimentCrossValidate


def boostingLearner(data, protectedIndex, protectedValue):
   h = boosting.boost(data)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)


def svmLearner(data, protectedIndex, protectedValue):
   h = svm.svmSKL(data, verbose=True)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)


def svmLinearLearner(data, protectedIndex, protectedValue):
   h = svm.svmSKL(data, kernel='linear', verbose=True)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)


def lrLearner(data, protectedIndex, protectedValue):
   h = lr.lrSKL(data)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)


@arrayErrorBars(2)
def statistics(train, test, protectedIndex, protectedValue, learner):
   h = learner(train, protectedIndex, protectedValue)
   print("Computing error")
   error = labelError(test, h)
   print("Computing bias")
   bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
   print("Computing UBIF")
   ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
   return error, bias, ubif


@errorBars(10)
def indFairnessStats(train, learner):
   print("Computing UBIF")
   ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
   return ubif


def runAll():
   print("Random Relabeling")
   experiments = [
      (('SVM', svmLearner), adult),
      (('SVMlinear', svmLinearLearner), german),
      (('SVM', svmLearner), singles),
      (('AdaBoost', boostingLearner), adult),
      (('AdaBoost', boostingLearner), german),
      (('AdaBoost', boostingLearner), singles),
      (('LR', lrLearner), adult),
      (('LR', lrLearner), german),
      (('LR', lrLearner), singles),
   ]

   for (learnerName, learner), dataset in experiments:
      print("%s %s" % (dataset.name, learnerName))
      experimentCrossValidate(dataset, learner, 5, statistics)


if __name__ == '__main__':
  runAll()
