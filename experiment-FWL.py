#!/usr/bin/env python3

import boosting
from utils import arrayErrorBars, variance
from weaklearners.decisionstump import buildDecisionStump
import errorfunctions as ef
from data import adult, singles, german
import random


def makeErrorFunction(protectedIndex, protectedValue, spWeight):
   '''
   Run boosting on a decision stump finder that uses an error function which
   is a linear combination of statistical imparity and label error

       w * statisticalParity + (1-w) * labelError
   '''
   sp = lambda data, h: ef.statisticalParity(data, protectedIndex, protectedValue, h=h)
   le = ef.minLabelErrorOfHypothesisAndNegation
   return ef.makeLinearCombination(sp, le, spWeight)


@arrayErrorBars(2)
def statistics(train, test, protectedIndex, protectedValue, numRounds=20):
   weight = 0.5
   flipProportion = 0.2

   error = makeErrorFunction(protectedIndex, protectedValue, weight)
   weakLearner = lambda draw: buildDecisionStump(draw, errorFunction=error)

   h = boosting.boost(train, weakLearner=weakLearner)

   bias = ef.signedStatisticalParity(test, protectedIndex, protectedValue, h)
   error = ef.labelError(test, h)
   ubif = ef.individualFairness(train, boosting.boost, flipProportion)

   return error, bias, ubif


def experimentCrossValidate(dataModule, times):
   PI = dataModule.protectedIndex
   PV = dataModule.protectedValue
   originalTrain, originalTest = dataModule.load()
   allData = originalTrain + originalTest

   variances = [[], [], []]  # error, bias, ubif
   mins = [float('inf'), float('inf'), float('inf')]
   maxes = [-float('inf'), -float('inf'), -float('inf')]
   avgs = [0, 0, 0]

   for time in range(times):
     random.shuffle(allData)
     train = allData[:len(originalTrain)]
     test = allData[len(originalTrain):]
     output = statistics(train, test, PI, PV)

     print("\tavg, min, max, variance")
     print("error: %r" % (output[0],))
     print("bias: %r" % (output[1],))
     print("ubif: %r" % (output[2],))

     for i in range(len(output)):
        avgs[i] += (output[i][0] - avgs[i]) / (time + 1)
        mins[i] = min(mins[i], output[i][1])
        maxes[i] = max(maxes[i], output[i][2])
        variances[i].append(output[i][0])  # was too lazy to implement online alg
        # warning: this doesn't take into account the variance of each split

   for i in range(len(variances)):
     variances[i] = variance(variances[i])

   print("AGGREGATE STATISTICS:")
   print("\tavg, min, max, variance")
   print("error: %r" % ((avgs[0], mins[0], maxes[0], variances[0]),))
   print("bias: %r" % ((avgs[1], mins[1], maxes[1], variances[1]),))
   print("ubif: %r" % ((avgs[2], mins[2], maxes[2], variances[2]),))


def runAll():
   datasets = [
      adult,
      singles,
      german,
   ]

   print("Fair weak learner")
   for dataset in datasets:
      experimentCrossValidate(dataset, 5)


if __name__ == '__main__':
   runAll()
