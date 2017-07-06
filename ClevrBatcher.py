import numpy as np
from pdb import set_trace as T
import os
import json
import h5py
import time

from lib import utils
from lib import nlp
from model.Tree import BTree

class ClevrBatcher():
   def __init__(self, batchSize, split, maxSamples=None, rand=True):
      dat = h5py.File('data/preprocessed/clevr.h5', 'r')
      self.questions = dat[split + 'Questions']
      self.answers   = dat[split + 'Answers']
      self.programs  = dat[split + 'Programs']
      self.imgs    = dat[split + 'Imgs']

      self.pMask     = dat[split + 'ProgramMask']
      self.imgIdx    = dat[split + 'ImageIdx']

      self.batchSize = batchSize
      if maxSamples is not None: 
         self.m = maxSamples
      else:
         self.m = len(self.questions)//batchSize*batchSize
      self.batches = self.m // batchSize
      self.pos = 0
 
   def next(self):
      batchSize = self.batchSize
      if (self.pos + batchSize) > self.m:
         self.pos = 0

      #Hack to fix stupid h5py indexing bug
      imgIdx    = self.imgIdx[self.pos:self.pos+batchSize]
      uniqueIdx = np.unique(imgIdx).tolist()
      mapTo = np.arange(len(uniqueIdx)).tolist()
      mapDict = dict(zip(uniqueIdx, mapTo))
      relIdx = [mapDict[x] for x in imgIdx]

      imgs      = self.imgs[np.unique(imgIdx).tolist()][relIdx] #Hack to fix h5py unique indexing bug
      questions = self.questions[self.pos:self.pos+batchSize]
      answers   = self.answers[self.pos:self.pos+batchSize]
      programs  = self.programs[self.pos:self.pos+batchSize]
      
      pMask     = self.pMask[self.pos:self.pos+batchSize]

      self.pos += batchSize
      return [questions, imgs, imgIdx], [programs, answers], [pMask]
