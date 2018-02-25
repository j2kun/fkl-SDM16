from data import adult, german, singles
from margin import svmRBFMarginAnalyzer, boostingMarginAnalyzer, lrSKLMarginAnalyzer


dataModules = [adult, german, singles]
methods = [
   ('svm', svmRBFMarginAnalyzer),
   ('lr', lrSKLMarginAnalyzer),
   ('boosting', boostingMarginAnalyzer)
]

for dataset in dataModules:
   for methodName, marginAnalyzer in methods:
      filenameBase = '%s-%s' % (dataset.name, methodName)
      print('Loading ' + dataset.name)
      tr, te = dataset.load()
      print('Training and computing %s margins' % methodName)
      ma = marginAnalyzer(tr, dataset.protectedIndex, dataset.protectedValue)
      print('Plotting histogram')
      ma.plotMarginHistogram(filename='plots/margin-histograms/%s-MH.pdf' % filenameBase)
      print('Plotting tradeoff')

      ma.plotTradeoff(filename='plots/tradeoffs/%s-T.pdf' % filenameBase)
      print('-' * 80)
