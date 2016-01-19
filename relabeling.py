from errorfunctions import signedStatisticalParity
from utils import sign, zeroOneSign, median
from random import random
from boosting import absMargin, margin
import math


# return a new hypothesis which flips the label of a data point if it is in the
# protected class and its boosting margin is less than the given threshold in
# magnitude
def thresholdRelabel(h, trainingData, protectedIndex, protectedValue,
                     hypotheses, weights, threshold):
   bias = signedStatisticalParity(trainingData, protectedIndex, protectedValue, h)
   biasedClass = 1 - zeroOneSign(bias)

   def relabel(pt):
      proposedLabel = h(pt)
      if (pt[protectedIndex] == biasedClass and
          absMargin(pt, hypotheses, weights) < threshold):
         return -proposedLabel
      else:
         return proposedLabel

   return relabel


#randomly flips labels of input classifier to kill bias of feature at index proteted_feature_index
#outputs the modified classifier
#only chooses labels that are on the 'non-favored' side of the feature that were rated -1
# to get rated 1
def randomOneSideRelabelData(h, trainingData, protectedIndex, protectedValue):
   bias = signedStatisticalParity(trainingData, protectedIndex, protectedValue, h)
   favored_trait = zeroOneSign(bias)

   nonfavored_data = [(feats,label) for feats,label in trainingData if not feats[protectedIndex]==favored_trait]
   NF, NFn = len(nonfavored_data), len([1 for x,label in nonfavored_data if h(x)==-1])

   p = NF*abs(bias)/NFn
   def relabeledClassifier(point):
      origClass = h(point)
      if point[protectedIndex] != favored_trait and origClass == -1:
         if random() < p:
            return -origClass
         else:
            return origClass
      else:
         return origClass

   return relabeledClassifier

if __name__ == '__main__':
   from data import adult
   from boosting import boost
   trainingData, testData = adult.load()
   protectedIndex = adult.protectedIndex
   protectedValue = adult.protectedValue

   h = boost(trainingData, 5)
   rr = randomOneSideRelabelData(h, trainingData, protectedIndex, protectedValue)
