import numpy as np
from scipy.ndimage import imread
from scipy.ndimage import zoom
from pdb import set_trace as T
import os
import json
import h5py

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

      #Hack to fix stupic h5py indexing bug
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

'''
#Read data
with open('data/questions/CLEVR_'+split+'_questions.json') as dataF:
   questions = json.load(dataF)['questions'][:20]
   questions = [*map(lambda e: (e['image_filename'], e['question'], e['answer'], e['program']), questions)]
   fNames, questions, answers, programs = [*zip(*questions)]
if split != 'test':
   with open('data/scenes/CLEVR_'+split+'_scenes.json') as dataF:
      scenes = json.load(dataF)['scenes']


#Setup images
imgNames =  sorted(os.listdir('data/images/'+split))
imgs = [imread('data/images/'+split+'/'+e) for e in imgNames[:26]]
imgInds  = [int(e[12:-4]) for e in fNames]

imgs = np.stack(imgs).astype(np.float32)/255.0
#imgs = zoom(imgs, (1, 0.7, 0.7, 1))[:, :, 56:-56, :]

#Preprocess programs
pVocab = nlp.buildVocab('PVocab.txt', word=True)
progs = []
for p in programs:
   p = BTree(p).flat()
   pNew = []
   for e in p:
      pNew += [pVocab[0][e]]
   p = pNew
   p = [p + (45-len(p))*[0]]
   progs += p
programs = np.asarray(progs).astype(np.int)

#Apply vocab
qVocab, qInvVocab  = nlp.buildVocab('XVocab.txt', word=True)
aVocab, aInvVocab  = nlp.buildVocab('YVocab.txt', word=True)
questions = [nlp.applyVocab(e[:-1].lower().split()+ ['?'], qVocab) for e in questions] 
answers = np.asarray([nlp.applyVocab(e.split(), aVocab) for e in answers])

#imgInds = np.random.randint(0,100,100)
#imgs = np.random.rand(100, 3, 224, 224).astype(np.float32)
#questions = np.random.randint(0,50,(100,8))
#answers = np.random.randint(0,60,100)
#programs = np.random.randint(0, 40, (100, 45))

self.imgs=imgs
#self.imgs = imgs.transpose(0, 3, 1, 2)
self.imgInds = imgInds
self.questions = questions
self.answers = answers
self.programs = programs

self.rand = rand
self.m = len(self.questions)
self.pos = 0
if self.rand:
   mask = np.random.permutation(np.arange(self.m))
   self.imgInds = np.asarray([self.imgInds[ind] for ind in mask])
   self.questions = [self.questions[ind] for ind in mask]
   self.answers = np.asarray([self.answers[ind] for ind in mask])
 
#Pad questions
newQuestions = []
for q in self.questions:
   newQuestions += [q.tolist() + [0]*(45-len(q))]
self.questions = np.asarray(newQuestions)

'''
