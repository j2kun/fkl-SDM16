#!/usr/bin/env python3
from random import random, sample
import sys
import math

from boosting import boost, adaboostGenerator, weightedLabelError
from weaklearners.decisionstump import buildDecisionStump
from errorfunctions import signedStatisticalParity, labelError
from utils import draw, normalize, sign, zeroOneSign

# randomly flips labels of the examples to kill bias of feature at protectedIndex
# only chooses labels that are on the 'non-favored' side of the feature that were rated -1
# to get rated 1
def randomOneSideMassageData(examples, protectedIndex, protectedValue):
   bias = signedStatisticalParity(examples, protectedIndex, protectedValue)
   print("Initial bias:", bias)
   favored_trait = 1-protectedValue

   #break up data by label and by the value of the protected trait
   favored_data = [(x,label) for x,label in examples if x[protectedIndex]==favored_trait]
   nonfavored_data = [(x,label) for x,label in examples if x[protectedIndex]!=favored_trait]
   favored_data_positive = [pt for pt in favored_data if pt[1]==1]
   nonfavored_data_negative = [pt for pt in nonfavored_data if pt[1]==-1]

   print("len(favored_data): %.3f" % len(favored_data))
   print("len(nonfavored_data): %.3f" % len(nonfavored_data))
   print("len(favored_data_positive): %.3f" % len(favored_data_positive))
   print("len(nonfavored_data_negative): %.3f" % len(nonfavored_data_negative))

   #calculate number of labels to flip from -1 to +1 on the nonfavored side
   num_nonfavored_positive = len(nonfavored_data)-len(nonfavored_data_negative)
   print("len(num_nonfavored_positive): %.3f" % num_nonfavored_positive)
   num_to_flip = math.floor((len(nonfavored_data)*len(favored_data_positive) - len(favored_data)*num_nonfavored_positive)/len(favored_data))
   print("Number of labels flipped:", num_to_flip)

   to_flip_to_pos = sample(nonfavored_data_negative, num_to_flip)

   flipped_examples = []
   for data in examples:
      if data in to_flip_to_pos:
         flipped_examples.append((data[0],-1*data[1]))
      else:
         flipped_examples.append(data)

   return flipped_examples

