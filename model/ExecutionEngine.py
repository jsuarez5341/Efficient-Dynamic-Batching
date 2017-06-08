from pdb import set_trace as T
import numpy as np
import time

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from model.ResNetTrunc import ResNet
from model.Tree import BTree
from model.Program import Program
from model.Program import Executioner
from model.Program import FastExecutioner
from model.Program import FasterExecutioner
from lib import utils

class ExecutionEngine(nn.Module):
   def __init__(self, 
         numUnary, numBinary, numClasses):
      super(ExecutionEngine, self).__init__()
      self.arities = 2*[0] + [1]*numUnary+[2]*numBinary
      unaries = [UnaryModule() for i in range(numUnary)]
      binaries = [BinaryModule() for i in range(numBinary)]
      self.cells = nn.ModuleList(2*[None] + unaries + binaries)
      self.CNN = CNN()
      self.classifier = EngineClassifier(numClasses)

   def forward(self, x, trainable=None):
      p, img = x
      a = []
      
      imgFeats = self.CNN(img)
      p = p.data.cpu().numpy().tolist()

      fast = True
      if fast:
         #Can parallelize
         progs = []
         for i in range(len(p)):
            pi = p[i]
            feats = imgFeats[i:i+1]
            prog = Program(pi, feats, self.arities)
            prog.build()
            progs += [prog]
         exeQ = FasterExecutioner(progs, self.cells)
         a = exeQ.execute()
         a = self.classifier(a)
      else:  
         #Can't parallelize
         progs = []
         a = []
         execs = []
         for i in range(len(p)):
            pi = p[i]
            feats = imgFeats[i:i+1]
            prog = Program(pi, feats, self.arities)
            prog.build()
            exeQ = Executioner(prog, self.cells)
            a += [exeQ.execute()]
            execs += [exeQ]
         a = t.cat(a, 0)
         a = self.classifier(a)
         
      return a

class EngineClassifier(nn.Module):
   def __init__(self, numClasses):
      super(EngineClassifier, self).__init__()
      self.conv1  = utils.Conv2d(128, 512, 1)
      self.fc1    = nn.Linear(512 * 7 * 7, 1024)
      self.pool   = nn.MaxPool2d(2)
      self.fc2    = nn.Linear(1024, numClasses)

   def forward(self, x) :
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = x.view(x.size()[0], -1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

class CNN(nn.Module):
   def __init__(self):
      super(CNN, self).__init__()
      self.conv1  = utils.Conv2d(1024, 128, 3)
      self.conv2  = utils.Conv2d(128, 128, 3)

   def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      return x

class UnaryModule(nn.Module):
   def __init__(self):
      super(UnaryModule, self).__init__()
      self.conv1  = utils.Conv2d(128, 128, 3)
      self.conv2  = utils.Conv2d(128, 128, 3)

   def forward(self, x):
      inp = x
      x = F.relu(self.conv1(x))
      x = self.conv2(x)
      x += inp
      x = F.relu(x)
      return x

class BinaryModule(nn.Module):
   def __init__(self):
      super(BinaryModule, self).__init__()
      self.conv1  = utils.Conv2d(256, 128, 1)
      self.conv2  = utils.Conv2d(128, 128, 3)
      self.conv3  = utils.Conv2d(128, 128, 3)

   def forward(self, x1, x2):
      x = t.cat((x1, x2), 1)
      x = F.relu(self.conv1(x))
      res = x
      x = F.relu(self.conv2(x))
      x = self.conv3(x)
      x += res
      x = F.relu(x)
      return x

