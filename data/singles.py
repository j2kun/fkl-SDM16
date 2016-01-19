from .datautils import *

name = "singles"
protectedIndex = 0 # gender
protectedValue = 1 # female

def toInt(val):
   if val.isdigit():
      return int(val)
   else:
      return -1

def processLine(line):
   values = line.strip().split(',')
   (income, sex, marital, age, educ, occup, resid, dualInc, perInHou,
    under18, homeStatus, homeType, ethnic, language) = values

   # shift sex by -1 to get it binary.
   point = [toInt(sex)-1, toInt(ethnic), toInt(age), toInt(educ), toInt(occup),
         toInt(resid), toInt(perInHou), toInt(under18), toInt(homeStatus),
         toInt(homeType), toInt(language)]

   label = 1 if int(income[0]) >= 5 else -1

   return tuple(point), label


def column(A,j):
   return [row[j] for row in A]

def transpose(A):
   return [column(A,j) for j in range(len(A[0]))]

def normalize(L):
   theMin = min(L)
   theMax = max(L)

   if theMax == 1 and theMin == 0 or theMax == theMin:
      return L

   return [(x - theMin) / (theMax - theMin) for x in L]


def normalizeExamples(data):
   points, labels = zip(*data)

   points = transpose([normalize(row) for row in transpose(points)])

   return list(zip(points, labels))


def load(normalize=False):
   trainFilename, testFilename = datasetFilenames('singles')

   with open(trainFilename, 'r') as infile:
      trainingData = [processLine(line) for line in infile]

   with open(testFilename, 'r') as infile:
      testData = [processLine(line) for line in infile]

   if normalize:
      return normalizeExamples(trainingData), normalizeExamples(testData)


   return trainingData, testData


