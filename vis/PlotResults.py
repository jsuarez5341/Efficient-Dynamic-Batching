from pdb import set_trace as T
import numpy as np
from matplotlib import pyplot as plt


def plotResults():
   #A bloody mess
   batch = [1, 32, 64, 320, 640, 850]
   fVanilla = [0.0031771, 0.0031694, 0.0026328, 0.00238375, 0.0023333]
   fOurs = [0.003963, 0.00248858, 0.001686116, 0.000710902, 0.0005151, 0.00042235]
   cVanilla = [0.002315934, 0.00287098, 0.00249189, 0.002322, 0.002199]
   cOurs = [0.002244463, 0.00155909, 0.00117287, 0.000341281, 0.000202705, 0.000156266]


   lineWidth=3
   ls = 26
   fs = 24
   ts = 26
   leg = 24
   

   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.set_xscale('log', basex=2)
   ax.set_yscale('log', basey=2)
   ax.set_xlim(1, 1024)
   ax.set_ylim(2**-13, 2**-8)
   ax.tick_params(axis='x', labelsize=ls)
   ax.tick_params(axis='y', labelsize=ls)

   plt.xlabel('Minibatch Size', fontsize=fs)
   plt.ylabel('Execution Time (sec / example)', fontsize=fs)
   plt.title('Efficiency Gains with Improved Topological Sort', fontsize=ts)

   ax.hold(True)
   ax.plot(batch[:-1], fVanilla, LineWidth=lineWidth, label='Vanilla Forward')
   ax.plot(batch, fOurs, LineWidth=lineWidth, label='Our Forward')
   ax.plot(batch[:-1], cVanilla, LineWidth=lineWidth, label='Vanilla Cell')
   ax.plot(batch, cOurs, LineWidth=lineWidth, label='Our Cell')
   ax.legend(loc='lower left', shadow=False, prop={'size':leg})
   plt.show()

if __name__ == '__main__':
   plotResults()
