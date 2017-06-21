from pdb import set_trace as T
import time
import numpy as np
import torch
import torch as t
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from matplotlib import pyplot as plt
from matplotlib import ticker

from lib import utils 

#Used for each expert
class TwoLayerNet(nn.Module):
   def __init__(self, inp, hid, out):
      super(TwoLayerNet, self).__init__()
      self.fc1 = nn.Linear(inp, hid)
      self.fc2 = nn.Linear(hid, out)

   def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return x

#Sparsely gated mixture of experts
class MOE(nn.Module):
   
   def __init__(self, numExperts, k, dim, hid):
      super(MOE, self).__init__()
      #Gate weights
      self.Wg = nn.Linear(dim, numExperts)
      self.Wn = nn.Linear(dim, numExperts)

      self.experts = utils.list(TwoLayerNet, dim, hid, dim, n=numExperts)
      self.k = k

   #As in original paper
   def H(self, x):
      Wg = self.Wg(x)
      noise = Variable(t.randn(*Wg.size()))
      if Wg.data.type() == 'torch.cuda.FloatTensor':
         noise = noise.cuda()
      Wn = F.softplus(self.Wn(x))
      return Wg + noise*Wn
   
   #Take top k from H
   def gate(self, x):
      a = self.H(x)
      topk, inds = a.topk(self.k)
      return F.softmax(topk), inds

   #Naive implementation with loops
   def vanillaExperts(self, x, gates, expertInds):
      cellTime = 0.0
      ret = []
      for samp in range(x.size(0)):
         sampSum = 0.0
         for j in range(self.k):
            expertInd = expertInds[samp, j].data[0]
            expert = self.experts[expertInd]
            g = gates[samp, j].data[0]
            cellStart = time.time()
            exp = g*expert(x[samp:samp+1])
            cellTime += time.time() - cellStart
            sampSum += exp
         ret += [sampSum]
      ret = t.cat(ret, 0)
      return ret, cellTime
 
   #Our implementation with dynamic batching
   def fastExperts(self, x, gates, expertInds):
      cellTime = 0.0
      samps = x.size(0)
      experts = {}

      #Accumulate data indices by expert
      for i in range(samps):
         for j in range(self.k):
            experts.setdefault(expertInds[i,j].data[0], []).append((i, j))

      out = {}
      for expInd, datInds in experts.items():
         expert = self.experts[expInd]
         dat = [x[ind[0]:ind[0]+1] for ind in datInds]
         g = [gates[ind[0]:ind[0]+1, ind[1]:ind[1]+1] for ind in datInds]

         #Aggregate data per expert 
         g   = t.cat(g, 0)
         dat = t.cat(dat, 0)
 
         #Execute and time that particular expert
         cellStart = time.time()
         exp = expert(dat)
         exp = exp * g.expand_as(exp)
         cellTime += time.time() - cellStart

         #Replace in original order
         for ind in range(len(datInds)):
            i, j = datInds[ind]
            out.setdefault(i, []).append(exp[ind:ind+1])

      #Compute sums over data
      ret = []
      for i in range(samps):
         ret += [sum(out[i])]
      ret = t.cat(ret, 0)
      return ret, cellTime

   def forward(self, x, fast=False, unitTest=False):
      start = time.time()

      #Run gate
      gates, expertInds = self.gate(x)

      #Run experts
      if unitTest:
         vanilla, _ = self.vanillaExperts(x, gates, expertInds)
         fast, _    = self.fastExperts(x, gates, expertInds)
         return t.abs(vanilla - fast)
      elif fast:
         ret, cellTime = self.fastExperts(x, gates, expertInds)
      else:
         ret, cellTime = self.vanillaExperts(x, gates, expertInds)

      forwardTime = time.time() - start
      return ret, forwardTime, cellTime

def data(m, dim):
   return 0.01*np.random.randn(m, dim).astype(np.float32)

#Time forward/cell
def runTest(net, dat, fast):
   numTests = 1 #faster, and in practice, the numbers do not vary by much
   start = time.time()
   forwardTotal = 0.0
   cellTotal = 0.0
   for i in range(numTests):
      out, forwardTime, cellTime = net(dat, fast=fast)
      forwardTotal += forwardTime
      cellTotal += cellTime
   forwardTime = forwardTotal/numTests/dat.size(0)
   cellTime    = cellTotal/numTests/dat.size(0)
   return forwardTime, cellTime

#Check to ensure the fast and vanila versions perform
#the same computations. Don't forget to zero the noise!
def correctnessCheck(net, dat):
   eps = 1e-5
   diff = net(dat, unitTest=True)
   diff = (diff > eps).float()
   if t.sum(diff).data[0] > eps:
      return False
   return True

#Reproduce figs from paper
def prettyPlot(samps, dat, hid):
   fig, ax = plt.subplots()
   sz = 14
   plt.rc('xtick', labelsize=sz)
   plt.rc('ytick', labelsize=sz)

   ax.set_xticklabels([1]+samps, fontsize=sz)
   ax.set_yticklabels([1]+samps[::-1], fontsize=sz)

   ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
   ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

   ax.set_xlabel('Number of Experts', fontsize=sz)
   ax.set_ylabel('Minibatch Size', fontsize=sz)
   ax.set_title('MOE Cell Speedup Factor (hidden='+str(hid)+')', fontsize=sz+2)

   #Show cell values
   for i in range(len(samps)):
      for j in range(len(samps)):
         ax.text(i, j, str(dat[i,j])[:4], ha='center', va='center', fontsize=sz, color='gray')

   plt.imshow(cellTimes, cmap='viridis')
   plt.show()
 

if __name__ == '__main__':
   cuda=True
   dim = 10
   hid = 1000
   k = 1
   numExperts = 100
   m = 1000
   
   dat = data(m, dim)
   dat = utils.var(dat, volatile=False, cuda=cuda)

   moe = MOE(numExperts, k, dim, hid)
   if cuda:
      moe = moe.cuda()
   print('Running MOE tests...')
   isCorrect = correctnessCheck(moe, dat)
   print('Test status: ', isCorrect)

   samps = [1,10,100,1000,10000]
   expSamps, mSamps = np.meshgrid(samps, samps)
   mSamps = np.flipud(mSamps)
   forwardTimes = np.zeros_like(expSamps).astype(np.float32)
   cellTimes = np.zeros_like(expSamps).astype(np.float32)
   
   #Produce data for figs
   sz = len(samps)
   for i in range(sz):
      for j in range(sz): 
         print('Tick')
         m = int(mSamps[i,j])
         exp = int(expSamps[i,j])
         dat = data(m, dim)
         dat = utils.var(dat, volatile=False, cuda=cuda)
         moe = MOE(exp, k, dim, hid)
         if cuda:
            moe = moe.cuda()
 
         forwardVanilla, cellVanilla = runTest(moe, dat, fast=False)
         forwardFast, cellFast       = runTest(moe, dat, fast=True)

         forwardTime = forwardVanilla / forwardFast
         cellTime    = cellVanilla / cellFast
         forwardTimes[i, j] = forwardTime
         cellTimes[i, j] = cellTime

   np.save('vis/cellspeed.npy', cellTimes)
   np.save('vis/forwardspeed.npy', forwardTimes)

   #Fig from paper
   prettyPlot(samps, cellTimes, hid)
     
