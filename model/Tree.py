from pdb import set_trace as T
from copy import deepcopy
import json

class Node():
   def __init__(self, cell):
      self.nxt = cell['inputs'][::-1]

      self.func = cell['function']
      if len(cell['value_inputs']) > 0:
         self.func += '_' + cell['value_inputs'][0]

class BTree():
   def __init__(self, cells):
      self.root = Node(cells[-1])
      self.addNodes(cells[:-1], self.root)

   def addNodes(self, cells, cur):
      for i in range(len(cur.nxt)):
         e = cur.nxt[i]
         node = Node(cells[e])
         cur.nxt[i] = node
         self.addNodes(cells, cur.nxt[i])

   def flat(self):
      return self.flatInternal(self.root, [])

   def flatInternal(self, cur, flattened):
      flattened += [cur.func]
      for e in cur.nxt:
         self.flatInternal(e, flattened)

      return flattened

   def print(self):
      self.printInternal(self.root)

   def printInternal(self, cur):
      print(cur.func)
      for e in cur.nxt:
         self.printInternal(e)


if __name__ == '__main__':
   dat = loadDat()
   q = dat[4]['program']
   tree = BTree(deepcopy(q))
   tree.print()
   T()

