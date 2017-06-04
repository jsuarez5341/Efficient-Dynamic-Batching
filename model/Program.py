from pdb import set_trace as T
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby
import time

class Node():
   def __init__(self, prev):
      self.prev = prev 
      self.inpData = []

   def build(self, cellInd, arity):
      self.next = [None] * arity
      self.arity = arity
      self.cellInd = cellInd

class Program:
   def __init__(self, prog, imgFeats, arities):
      self.prog = prog
      self.imgFeats = imgFeats
      self.arities = arities
      self.root = Node(None)

   def build(self, ind=0):
      self.buildInternal(self.root)

   def buildInternal(self, cur=None, count=0):
      if count >= len(self.prog):
         arity = 0
      else:
         ind = self.prog[count]
         arity = self.arities[ind]

      cur.build(ind, arity)

      if arity == 0:
         cur.inpData = [self.imgFeats]
      elif arity == 1:
         cur.next = [Node(cur)]
         count = self.buildInternal(cur.next[0], count+1)
      elif arity == 2:
         cur.next = [Node(cur), Node(cur)]
         count = self.buildInternal(cur.next[0], count+1)
         count = self.buildInternal(cur.next[1], count+1)

      return count

   def flat(self):
      return self.flatInternal(self.root, [])

   def flatInternal(self, cur, flattened):
      flattened += [cur.cellInd]
      for e in cur.next:
         self.flatInternal(e, flattened)

      return flattened

   def topologicalSort(self):
      return self.topInternal(self.root, [])

   def topInternal(self, cur, flattened):
      for e in cur.next:
         self.topInternal(e, flattened)

      flattened += [cur]
      return flattened

class HighArcESort:
   def __init__(self):
      self.out = {}

   def __call__(self, root, maxDepth):
      assert(not self.out) #Empty
      self.highArcESortInternal(root, maxDepth)
      return self.out

   def highArcESortInternal(self, cur, rank):
      for nxt in cur.next:
         ret = self.highArcESortInternal(nxt, rank)
         rank = min(rank, ret)
      self.out[rank] = cur
      return rank-1

class FasterExecutioner:
   def __init__(self, progs, cells):
      self.cells = cells

      self.progs = progs
      self.sortProgs()

   def sortProgs(self):
      maxDepth = 9001 #An arbitrary power level
      for i in range(len(self.progs)):
         self.progs[i] = HighArcESort()(self.progs[i].root, maxDepth)
   
   def execute(self):
      maxLen = max([len(e) for e in self.progs])
      for s in range(maxLen):
         nodes = []
         for i in range(len(self.progs)):
            prog = self.progs[i]
            if len(prog) <= s:
               continue
            nodes += [prog[s]]

         groupedNodes = {}
         for node in nodes:
            groupedNodes.setdefault(node.cellInd, []).append(node)

         for cellInd, nodes in groupedNodes.items():
            arity = nodes[0].arity
            cell = self.cells[cellInd]

            outData = [node.inpData[0] for node in nodes]
            if arity==1:
               arg = t.cat(outData, 0)
               outData = cell(arg)
               outData = t.split(outData, 1, 0)
            elif arity==2:
               arg1 = t.cat(outData, 0)
               arg2 = t.cat([node.inpData[1] for node in nodes], 0)
               outData = cell(arg1, arg2) 
               outData = t.split(outData, 1, 0)
            
            for node, outDat in zip(nodes, outData):
               if node.prev is None:
                  node.outData = outDat
               else:
                  node.prev.inpData += [outDat]

      outData = [prog[-1].outData for prog in self.progs]
      return t.cat(outData, 0)


class FastExecutioner:
   def __init__(self, progs, cells):
      self.cells = cells

      self.progs = progs
      self.sortProgs()

   def sortProgs(self):
      for i in range(len(self.progs)):
         self.progs[i] = self.progs[i].topologicalSort()
   
   def execute(self):
      maxLen = max([len(e) for e in self.progs])
      for s in range(maxLen):
         nodes = []
         for i in range(len(self.progs)):
            prog = self.progs[i]
            if len(prog) <= s:
               continue
            nodes += [prog[s]]

         groupedNodes = {}
         for node in nodes:
            groupedNodes.setdefault(node.cellInd, []).append(node)

         for cellInd, nodes in groupedNodes.items():
            arity = nodes[0].arity
            cell = self.cells[cellInd]

            outData = [node.inpData[0] for node in nodes]
            if arity==1:
               arg = t.cat(outData, 0)
               outData = cell(arg)
               outData = t.split(outData, 1, 0)
            elif arity==2:
               arg1 = t.cat(outData, 0)
               arg2 = t.cat([node.inpData[1] for node in nodes], 0)
               outData = cell(arg1, arg2) 
               outData = t.split(outData, 1, 0)
            
            for node, outDat in zip(nodes, outData):
               if node.prev is None:
                  node.outData = outDat
               else:
                  node.prev.inpData += [outDat]

      outData = [prog[-1].outData for prog in self.progs]
      return t.cat(outData, 0)

class Executioner:
   def __init__(self, prog, cells):
      self.prog = prog
      self.cells = cells

   def execute(self):
      return self.executeInternal(self.prog.root)

   def executeInternal(self, cur):
      if cur.arity == 0:
         return cur.inpData[0]
      elif cur.arity == 1:
         args = [self.executeInternal(cur.next[0])]
      elif cur.arity == 2:
         arg1 = self.executeInternal(cur.next[0])
         arg2 = self.executeInternal(cur.next[1])
         args = [arg1, arg2]

      cell = self.cells[cur.cellInd]
      return cell(*args)


'''
class Node():
   def __init__(self, cell, arity=None):
      self.nxt = [None] * arity
      self.func = cell

class Program:
   def __init__(self, prog, imgFeats, cells, arities):
      self.prog = prog
      self.imgFeats = imgFeats
      self.cells = cells
      self.arities = arities

   def execute(self):
      return self.executeInternal()[0]

   def executeInternal(self, ind=0):
      if ind >= len(self.prog):
         cur = self.cells[0]
         arity = 0
      else:
         cur = self.cells[self.prog[ind]]
         arity = self.arities[self.prog[ind]]

      if cur.arity == 0:
         return self.imgFeats, ind
      if cur.arity == 1:
         arg1, ind = self.executeInternal(ind+1)
         return cur(arg1), ind
      elif arity == 2:
         arg1, ind = self.executeInternal(ind+1)
         arg2, ind = self.executeInternal(ind+1)
         return cur(arg1, arg2), ind

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
'''

