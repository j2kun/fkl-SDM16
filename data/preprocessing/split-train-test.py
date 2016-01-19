#!/usr/bin/env python3

import random

def split(filename, splitFraction = 2/3):
   #datasetName, datasetType = filename.split('.')[0].split('-')
   datasetName = filename.split('.')[0]

   with open(filename, 'r') as infile:
      data = infile.readlines()[1:]

   random.shuffle(data)
   m = int(len(data) * splitFraction)
   trainingData = data[:m]
   testData = data[m:]

   #with open(datasetName + "-" + str(datasetType) + '.train', 'w') as outfile:
   with open(datasetName + '.train', 'w') as outfile:
      for line in trainingData:
         outfile.write(line)

   #with open(datasetName + "-" + str(datasetType) + '.test', 'w') as outfile:
   with open(datasetName + '.test', 'w') as outfile:
      for line in testData:
         outfile.write(line)


if __name__ == "__main__":
   #split('student-mat.csv', 2/3)
   #split('student-por.csv', 2/3)
   split('german.data', 2/3)
