import numpy as np
from pdb import set_trace as t
import os

def buildVocab(fName):
   dat = open(fName).read()
   chars = sorted(np.unique(list(dat)).tolist())
   vocab = dict(zip(list(chars), np.arange(len(chars))))
   invVocab = {v: k for k, v in vocab.items()}
   return vocab, invVocab 

def applyVocab(line, vocab):
   ret = []
   for e in line:
      ret += [vocab[e]]
   return np.asarray(ret)
      
class RawBatcher():
   def __init__(self, fName, vocabF, rand=True):
      self.rand = rand

      self.fLen = os.stat(fName).st_size
      self.dat = open(fName).read()
      self.m = len(self.dat)
      self.pos = 0

      self.vocab, self.invVocab = buildVocab(vocabF)
      self.dat = applyVocab(self.dat, self.vocab)

      #self.realDat = self.nextFake(1000, 515)

   def next(self, batchSize, seqLen, minContext):
      seqLen += 1 #One extra for label

      #Stupid, but stupidly fast. 
      #Other methods are smart and stupidly slow.     
      if self.rand:
         inds  = np.random.randint(0, self.fLen-seqLen, (batchSize, 1))
         inds = inds + (np.ones((batchSize, seqLen))*np.arange(seqLen)).astype(np.int)
      else:
         if (self.pos + batchSize*seqLen) > self.fLen:
            self.pos = 0 #Misses <1 minibatch. Don't be pedantic ;)
         inds = self.pos + np.arange(batchSize*(seqLen-minContext-1)).reshape(batchSize, seqLen-minContext-1)
         overlap = (inds[:, -1][:, np.newaxis] + np.arange(1, minContext+2) * np.ones((batchSize, minContext+1)))
         inds = np.hstack((inds, overlap)).astype(np.int)

      batch = self.dat[inds]

      X = batch[:, :-1]
      Y = batch[:, 1:]

      return X, Y

   #Debug function for testing runtime. Currently negligible
   def nextFake(self, batchSize, seqLen):
      X, Y = self.realDat
      return X[:batchSize], Y[:batchSize] 



