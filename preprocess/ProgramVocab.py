from pdb import set_trace as T
from copy import deepcopy
import json

import numpy as np


def loadDat():
   with open('../data/clevr/questions/CLEVR_val_questions.json') as dataF:
      questions = json.load(dataF)['questions']

   return questions

def getFuncs(dat):
   vocab = []
   for p in dat:
      p = p['program']
      for e in p:
         func = e['function']
         append = e['value_inputs']
         if len(append) > 0:
            func += '_' + append[0]

         func = str(len(e['inputs'])) + '_' + func
         vocab += [func]

   vocab = list(set(vocab))
   return sorted(vocab)

if __name__ == '__main__':
   dat = loadDat()
   vocab = getFuncs(dat)
   
   #0: 1
   #1: 30
   #2: 9
   with open('../data/vocab/ProgramVocab.txt', 'w') as f:
      for e in vocab:
         f.write(e+ ' ')
