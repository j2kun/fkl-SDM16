from .datautils import *

name = "student"
protectedIndex = 1

jobs = ('teacher', 'health', 'services', 'at_home', 'other')
reasons = ('home', 'reputation', 'course', 'other')
guardians = ('mother', 'father', 'other')

featureNames=('school','sex','age','address','famsize','Pstatus','Medu','Fedu','M:teacher','M:health','M:services','M:at_home','M:other','F:teacher','F:health','F:services','F:at_home','F:other','reason:home','reason:reputation','reason:course','reason:other','guardian:mother','guardian:father','guardian:other','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences')


def processLine(line):
   values = line.strip().split(',')
   (school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3)=values

   #average of G3 is 10.41, median is 11. The paper where this dataset was introduced used 10 as a threshold for passing.
   label = 1 if int(G3)>=10 else -1

   point = [int(i) for i in values[:8]]
   for (i,j) in zip((Mjob,Fjob,reason,guardian),(jobs,jobs,reasons,guardians)):
      point += vectorize(i,j)
   point += [int(i) for i in values[12:-3]]

   return tuple(point), label


def load(mathOrPortuguese, name = "student"):
   trainFilename, testFilename = datasetFilenames(name + '-' + mathOrPortuguese)

   with open(trainFilename, 'r') as infile:
      trainingData = [processLine(line) for line in infile]

   with open(testFilename, 'r') as infile:
      testData = [processLine(line) for line in infile]

   return trainingData, testData


def loadMath():
   load('mat')

def loadPortuguese():
   load('por')


