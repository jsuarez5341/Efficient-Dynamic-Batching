import numpy as np
import sys
from pdb import set_trace as T

import torch as t
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init

#Generic
def invertDict(x):
   return {v: k for k, v in x.items()}

def loadDict(fName):
   with open(fName) as f:
      s = eval(f.read())
   return s

def norm(x, n=2):
   return (np.sum(np.abs(x)**n)**(1.0/n)) / np.prod(x.shape)

#Continuous moving average
class CMA():
   def __init__(self):
      self.t = 0.0
      self.cma = 0.0
   
   def update(self, x):
      self.cma = (x + self.t*self.cma)/(self.t+1)
      self.t += 1.0

#Exponentially decaying average
class EDA():
   def __init__(self):
      self.k = 0.99
      self.eda = 0.0
   
   def update(self, x):
      #self.eda = self.eda * k / (x * (1-k))
      self.eda = (1-self.k)*x + self.k*self.eda

#Print model size
def modelSize(net): 
   params = 0 
   for e in net.parameters(): 
      params += np.prod(e.size()) 
   params = int(params/1000) 
   print("Network has ", params, "K params")  

#Same padded (odd k)
def Conv2d(fIn, fOut, k):
   pad = int((k-1)/2)
   return nn.Conv2d(fIn, fOut, k, padding=pad)

#ModuleList wrapper
def list(module, *args, n=1):
   return nn.ModuleList([module(*args) for i in range(n)])

#Full-network initialization wrapper
def initWeights(net, scheme='orthogonal'):
   print('Initializing weights. Warning: may overwrite sensitive bias parameters (e.g. batchnorm)')
   for e in net.parameters():
      if scheme == 'orthogonal':
         if len(e.size()) >= 2:
            init.orthogonal(e)
      elif scheme == 'normal':
         init.normal(e, std=1e-2)
      elif scheme == 'xavier':
         init.xavier_normal(e)

class SaveManager():
   def __init__(self, root):
      self.tl, self.ta, self.vl, self.va = [], [], [], []
      self.root = root
      self.stateDict = None 
      self.lock = False

   def update(self, net, tl, ta, vl, va):
      nan = np.isnan(sum([t.sum(e) for e in net.state_dict().values()]))
      if nan or self.lock:
         self.lock = True
         print('NaN in update. Locking. Call refresh() to reset')
         return

      if self.epoch() == 1 or vl < np.min(self.vl):
         self.stateDict = net.state_dict().copy()
         t.save(net.state_dict(), self.root+'weights')

      self.tl += [tl]; self.ta += [ta]
      self.vl += [vl]; self.va += [va]
 
      np.save(self.root + 'tl.npy', self.tl)
      np.save(self.root + 'ta.npy', self.ta)
      np.save(self.root + 'vl.npy', self.vl)
      np.save(self.root + 'va.npy', self.va)

   def load(self, net, raw=False):
      stateDict = t.load(self.root+'weights')
      self.stateDict = stateDict
      if not raw:
         net.load_state_dict(stateDict)
      self.tl = np.load(self.root + 'tl.npy').tolist()
      self.ta = np.load(self.root + 'ta.npy').tolist()
      self.vl = np.load(self.root + 'vl.npy').tolist()
      self.va = np.load(self.root + 'va.npy').tolist()

   def refresh(self, net):
      self.lock = False
      net.load_state_dict(self.stateDict)

   def epoch(self):
      return len(self.tl)+1

def runMinibatch(net, batcher, cuda=True, volatile=False, trainable=False):
   x, y = batcher.next()
   x = [Variable(t.from_numpy(e), volatile=volatile) for e in x]
   y = [Variable(t.from_numpy(e), volatile=volatile) for e in y]
   if cuda:
      x = [e.cuda() for e in x]
      y = [e.cuda() for e in y]

   if len(x) == 1:
      x = x[0]
   if len(y) == 1:
      y = y[0]

   a = net(x, trainable)
   return a, y

def timeGrads(net, cell, batcher, criterion=nn.CrossEntropyLoss(), cuda=True):   
   iters = batcher.batches
   def hook(module, grad_input, grad_output):
      try:
         hook.timeGrads += [grad_output]
      except:
         hook.timeGrads = []

   cell.register_backward_hook(hook)

   for i in range(iters):
      a, y = runMinibatch(net, batcher, cuda)
      a = a[:, -1]
      y = y[:, -1]
      m = y.size()[0]
      a, y = a.contiguous().view(m, -1), y.contiguous().view(-1)

      loss = criterion(a, y)
      loss.backward(retain_variables=True)

      return [e[0].cpu().data.numpy() for e in hook.timeGrads[::-1]]
   

def gradCheck(net, batcher, criterion=nn.CrossEntropyLoss(), cuda=True):
   iters = batcher.batches
   for i in range(iters):
      a, y = runMinibatch(net, batcher, cuda)

      m = y.size()[0] * y.size()[1]
      a, y = a.contiguous().view(m, -1), y.contiguous().view(-1)

      loss = criterion(a, y)
      loss.backward(retain_variables=True)
   
      keys = net.state_dict().keys()
      grads = [e for e in net.parameters()]

      #Got sick of functional
      out = {}
      keys = [e for e in net.state_dict().keys()]
      for i in range(len(grads)):
         out[keys[i]] = grads[i]

      return out
     
def runData(net, opt, batcher, criterion=nn.CrossEntropyLoss(), 
      trainable=False, verbose=False, cuda=True,
      gradClip=10.0, minContext=0):
   iters = batcher.batches
   meanAcc  = CMA()
   meanLoss = CMA()

   for i in range(iters):
      if verbose and i % int(iters/10) == 0:
         sys.stdout.write('#')
         sys.stdout.flush()

      a, y = runMinibatch(net, batcher, trainable=trainable, cuda=cuda, volatile=not trainable)

      m = np.prod(y.size()).item()
      a, y = a.view(m, -1), y.view(-1)
      m = t.sum(y != 0).data[0]

      loss = criterion(a, y)
      if trainable:
         opt.zero_grad()
         loss.backward()
         if gradClip is not None:
            t.nn.utils.clip_grad_norm(net.parameters(), 
                  gradClip, norm_type=1)
         opt.step()

      #Stats
      _, preds = t.max(a.data, 1)
      acc = sum((y.data == preds) * (y.data != 0)) / float(m)

      #Accumulate average
      meanLoss.update(loss.data[0])
      meanAcc.update(acc)

   return meanLoss.cma, meanAcc.cma


