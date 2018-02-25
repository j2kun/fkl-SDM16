from data import adult, german, singles
from errorfunctions import labelError, statisticalParity, signedStatisticalParity, individualFairness
from utils import arrayErrorBars, errorBars, experimentCrossValidate
from boosting import boost
from svm import svmSKL, svmLinearSKL
from lr import lrSKL

def boostingLearner(data, protectedIndex, protectedValue):
   return boost(data)

def svmLearner(data, protectedIndex, protectedValue):
   return svmSKL(data, verbose=True)

def svmLinearLearner(data, protectedIndex, protectedValue):
   return svmSKL(data, kernel='linear', verbose=True)

def lrLearner(data, protectedIndex, protectedValue):
   return lrSKL(data)

def generalStatistics(trainingData, testData, protectedIndex, protectedValue):
   print('Size of training data:', len(trainingData))
   print('Size of testing data:', len(testData))

   print("SP of training data: %f" % statisticalParity(trainingData, protectedIndex, protectedValue))
   print("SP of test data: %f" % statisticalParity(testData, protectedIndex, protectedValue))


FLIP_PROPORTION = 0.2

def runBaseline(trainingData, testData, learner, protectedIndex, protectedValue):
   h = learner(trainingData)

   print("Training error: %f" % labelError(trainingData, h))
   print("Test error: %f" % labelError(testData, h))
   print("SP of the hypothesis: %f" % statisticalParity(testData, protectedIndex, protectedValue, h))

   UBIF = individualFairness(trainingData, learner, FLIP_PROPORTION)
   print("UBIF of the hypothesis on training: %f" % UBIF)

@arrayErrorBars(2)
def statisticsForCV(train, test, protectedIndex, protectedValue, learner):
   h = learner(train, protectedIndex, protectedValue)
   print("Computing error")
   error = labelError(test, h)
   print("Computing bias")
   bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
   print("Computing UBIF")
   ubif = individualFairness(train, learner, 0.2, passProtected=True)
   return error, bias, ubif

@arrayErrorBars(20)
def statistics(trainingData, testData, learner, protectedIndex, protectedValue):
   h = learner(trainingData)

   trainingError = labelError(trainingData, h)
   testError = labelError(testData, h)
   sp = statisticalParity(testData, protectedIndex, protectedValue, h)
   UBIF = individualFairness(trainingData, learner, FLIP_PROPORTION)

   return trainingError, testError, sp, UBIF

@errorBars(10)
def indFairnessStats(trainingData, learner):
   UBIF = individualFairness(trainingData, learner, FLIP_PROPORTION)
   return UBIF


def runBaselineAveraged(train, test, learner, protectedIndex, protectedValue):
   output = statistics(train, test, learner, protectedIndex, protectedValue)
   print("\tavg, min, max, variance")
   print("train error: %r" % (output[0],))
   print("test error: %r" % (output[1],))
   print("bias: %r" % (output[2],))
   print("ubif: %r" % (output[3],))
   return output


def runAllGeneralStatistics():
  for dataset in (adult, german, singles):
    print('General statistics for ' + dataset.name)
    trainingData, testData = dataset.load()
    PI = dataset.protectedIndex
    PV = dataset.protectedValue
    generalStatistics(trainingData, testData, PI, PV)
    print('')

def runAllCrossValidate():
   print("Baseline algorithms")
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
      print("%s %s" % (dataset.name, learnerName), flush=True)
      experimentCrossValidate(dataset, learner, 5, statisticsForCV)

def runAll():
  runAllGeneralStatistics()
  runAllCrossValidate()

if __name__ == '__main__':
  runAll()
