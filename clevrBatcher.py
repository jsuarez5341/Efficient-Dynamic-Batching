import numpy as np
from scipy.ndimage import imread
from pdb import set_trace as T
import os
import json
import utils
import nlp

class ClevrBatcher():
   def __init__(self, split, rand=True):

      #Read data
      with open('data/questions/CLEVR_'+split+'_questions.json') as dataF:
         questions = json.load(dataF)['questions'][:256]
         questions = [*map(lambda e: (e['image_filename'], e['question'], e['answer']), questions)]
         fNames, questions, answers = [*zip(*questions)]
      if split != 'test':
         with open('data/scenes/CLEVR_'+split+'_scenes.json') as dataF:
            scenes = json.load(dataF)['scenes']


      #Setup images
      imgNames =  sorted(os.listdir('data/images/'+split))
      imgs = [imread('data/images/'+split+'/'+e) for e in imgNames[:26]]
      imgInds  = [int(e[12:-4]) for e in fNames]
      imgs = np.stack(imgs)

      #Apply vocab
      qVocab, qInvVocab  = nlp.buildVocab('XVocab.txt')
      aVocab, aInvVocab  = nlp.buildVocab('YVocab.txt')
      questions = [nlp.applyVocab(e[:-1].lower() + ' ?', qVocab) for e in questions] 
      answers = [nlp.applyVocab(e, aVocab) for e in answers] 
   
      #self.imgs = np.random.rand(100, 3, 224, 224)
      #self.questions = np.random.randint(0,50,(100,8))
      #self.answers = np.random.randint(0,60,100)

      self.imgs = imgs.transpose(0, 3, 1, 2)
      self.imgInds = imgInds
      self.questions = questions
      self.answers = answers

      self.rand = rand
      self.m = len(self.questions)
      self.pos = 0
      if self.rand:
         mask = np.random.permutation(np.arange(self.m))
         self.imgInds = np.asarray([self.imgInds[ind] for ind in mask])
         self.questions = [self.questions[ind] for ind in mask]
         self.answers = [self.answers[ind] for ind in mask]
       
      #Pad questions
      newQuestions = []
      for q in self.questions:
         newQuestions += [q.tolist() + [0]*(190-len(q))]
      self.questions = np.asarray(newQuestions)

   def next(self, batchSize):
      #Stupid, but stupidly fast. 
      #Other methods are smart and stupidly slow.     
      if (self.pos + batchSize) > self.m:
         self.pos = 0

      imgInds   = self.imgInds[self.pos:self.pos+batchSize]
      imgs      = self.imgs[imgInds]
      questions = self.questions[self.pos:self.pos+batchSize]
      answers   = self.answers[self.pos:self.pos+batchSize]
      self.pos += batchSize

      return imgs, questions, answers

