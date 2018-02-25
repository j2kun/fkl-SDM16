import pickle
import matplotlib.pyplot as plt

'''
Plot the margins for general class and protected class so we can figure out what
kind of relabeling is the right kind.
'''


def plot(margins, protectedIndex, protectedClass):
   marginList = list(margins.values())
   protectedMargins = [v for (x, v) in margins.items() if x[0][protectedIndex] == protectedClass]
   incorrectMargins = [v for (x, v) in margins.items() if v * x[1] < 0]
   incorrectProtectedMargins = [
      v for (x, v) in margins.items() if v * x[1] < 0 and x[0][protectedIndex] == protectedClass]

   f, (ax1, ax2) = plt.subplots(2, 1)

   # distribution of signed margins on test data
   ax1.hist(marginList, bins=40, label='population')
   ax1.hist(protectedMargins, bins=40, label='protected', color='y')
   ax1.set_xlim([-1, 1])
   ax1.set_title("Confidence Values of All Examples")

   ax2.hist(incorrectMargins, bins=40, label='population')
   ax2.hist(incorrectProtectedMargins, bins=40, label='protected', color='y')
   ax2.set_xlim([-1, 1])
   ax2.set_title("Confidence Values of Mislabeled Training Examples")

   plt.subplots_adjust(hspace=.75)
   plt.legend()
   plt.show()


if __name__ == '__main__':
   with open("results/baselines/margins.pickle", 'rb') as infile:
      margins = pickle.load(infile)

   plot(margins, 1, 0)
