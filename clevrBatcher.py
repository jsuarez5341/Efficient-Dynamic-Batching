import numpy as np
from scipy.ndimage import imread
from pdb import set_trace as t
import os
import json

def vocab(answers):
   v = np.unique(answers)
   vDict = dict(zip(v, np.arange(len(v))))
   return vDict

class ClevrBatcher():
   def __init__(self, split, rand=True):
      '''
      with open('data/questions/CLEVR_'+split+'_questions.json') as dataF:
         questions = json.load(dataF)['questions'][:50]
         questions = [*map(lambda e: (e['image_filename'], e['question'], e['answer']), questions)]
         fNames, questions, answers = [*zip(*questions)]
      if split != 'test':
         with open('data/scenes/CLEVR_'+split+'_scenes.json') as dataF:
            scenes = json.load(dataF)['scenes']

      imgs = [imread('data/images/'+split+'/'+f, mode='RGB') for f in fNames]
      imgs = np.stack(imgs)
      '''
      self.imgs = np.random.rand(100, 240, 240, 3)
      self.questions = np.random.randint(0,100,(50,20))
      self.answers = np.random.randint(0,50,50)
      self.rand = rand

      self.m = self.imgs.shape[0]
      self.pos = 0

   def next(self, batchSize):
      #Stupid, but stupidly fast. 
      #Other methods are smart and stupidly slow.     
      if (self.pos + batchSize) > self.m:
         self.pos = 0
      imgs      = self.imgs[self.pos:self.pos+batchSize]
      questions = self.questions[self.pos:self.pos+batchSize]
      answers   = self.answers[self.pos:self.pos+batchSize]
      self.pos += batchSize

      return imgs, questions, answers

