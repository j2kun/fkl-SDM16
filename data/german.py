from data.datautils import *
from itertools import permutations

name = "german"
protectedIndex = 0
protectedValue = 0

featureNames=None


def A2num(s):
   assert s[0]=='A'
   return int(s[1:])

def Arange(a,b):
   return ['A'+str(i) for i in range(a,b+1)]

def processLine(line):
   values = line.strip().split(' ')
   (checking,duration,history,purpose,amount,savings,employment,installment,marital,otherdg,residence,prop,age,otheri,housing,credits,job,maintenance,phone,foreign,risk)=values

   label = 1 if risk=='1' else -1

   point = (1 if int(age)>=25 else 0, A2num(checking) if checking != 'A14' else 0, int(duration), A2num(history),)+tuple(vectorize(purpose, Arange(40,49)+['A410']))+(int(amount), A2num(savings) if savings != 'A65' else 0, A2num(employment), int(installment))+tuple(vectorize(marital, Arange(91,95)))+(A2num(otherdg), int(residence), A2num(prop), int(age), A2num(otheri), A2num(housing), int(credits), A2num(job), int(maintenance), A2num(phone), A2num(foreign))

   return point, label


def load(name = "german"):
   trainFilename, testFilename = datasetFilenames(name)

   with open(trainFilename, 'r') as infile:
      trainingData = [processLine(line) for line in infile]

   with open(testFilename, 'r') as infile:
      testData = [processLine(line) for line in infile]

   return trainingData, testData

#generator that splits the whole data set into split number 
# of chunks, and yields each permutation of those chunks
def loadSplit(name = "german", split=3):
   datasetFilename = datasetFilenameAll(name)
   with open(trainFilename, 'r') as infile:
      data = [processLine(line) for line in infile]

   splitL = int(len(data)/split)
   chunks = [data[splitL*i : splitL*(i+1)] for i in range(split-1)] + [data[(split-1)*splitL:]]
   for dataChunks in permutations(chunks, split):
      yield dataChunks
   



