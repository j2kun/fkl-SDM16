import matplotlib.pyplot as plt
import numpy as np

def plot(shiftRelabel):
   xs, srBias, srError = np.array(shiftRelabel).T
   #xs2, trError, trBias = np.array(thresholdRelabel).T

   #f, (ax1, ax2) = plt.subplots(2,1)
   width = 3

   #plt.plot(figsize=(plt.figaspect(2.0)))

   plt.plot(xs, srError, label='Label error', linewidth=width)
   plt.plot(xs, srBias, label='Bias', linewidth=width)
   plt.title("Shifted Decision Boundary Bias vs Error")
   plt.gca().invert_xaxis()
   plt.axhline(0, color='black')
   plt.figaspect(10.0)
   #handles, labels = plt.get_legend_handles_labels()

   # ax2.plot(xs2, trError, label='Label error', linewidth=width)
   # ax2.plot(xs2, trBias, label='Bias', linewidth=width)
   # ax2.set_title("Margin Threshold Relabeling Bias vs Error")
   # ax2.axhline(0, color='black')

   #plt.subplots_adjust(hspace=.75)
   #plt.figlegend(handles, labels, 'center right')
   plt.legend(loc='center right')
   #plt.savefig("relabeling-msr-tradeoffs.pdf",bbox_inches='tight')
   plt.show()


if __name__ == '__main__':
   import csv
   with open("results/relabeling/margin-shift-relabeling.csv", 'r') as infile:
      shiftRelabel = [[float(x) for x in line] for line in list(csv.reader(infile))[1:]]
   with open("results/relabeling/threshold-relabeling.csv", 'r') as infile:
      thresholdRelabel = [[float(x) for x in line] for line in list(csv.reader(infile))[1:]]

   plot(shiftRelabel)
