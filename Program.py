from pdb import set_trace as T
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Node():
   def __init__(self, ind, arities, cells):
      self.arity = arities[ind]
      self.ann   = cells[ind]
      self.next = [None]*self.arity

class Program():
   def __init__(self, arities, cells):
      self.arities = arities
      self.cells = cells
      self.root = None 

   #Ind is the output argmax index of the prog generator
   def addNode(self, ind, cur=None):
      if self.root is None:
         self.root = Node(ind, self.arities, self.cells)
         return

      if cur is None:
         cur = self.root

      for i in range(cur.arity):
         #Branching wrong here
         if cur.next[i] is not None:
            if self.addNode(ind, cur.next[i]):
               return True
         else:
            cur.next[i] = Node(ind, self.arities, self.cells)
            return True
      return False

   def build(self, inds):
      for e in inds:
         self.addNode(e)

   def execute(self, imgFeats):
      return self.executeInternal(imgFeats, cur=self.root)

   def executeInternal(self, imgFeats, cur=None):
      if cur is None or cur.arity == 0:
         return imgFeats

      inps = []
      for i in range(cur.arity):
         inps += [self.executeInternal(imgFeats, cur.next[i])]

      return cur.ann(*inps)

   def print(self):
      self.printInternal(cur=self.root)

   def printInternal(self, cur=None, numTabs=0):
      if cur is None or cur.arity == 0: #Scene token
         print('\t'*numTabs+'SCENE')
         return
      else:
         print('\t'*numTabs,cur.ann)

      for i in range(cur.arity):
         self.printInternal(cur.next[i], numTabs+1)

