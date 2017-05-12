import numpy as np
import json
from pdb import set_trace as T

#Extract vocab from json
def getAllWords(fName):
   dat = open(fName).read()
   dat = json.loads(dat)
   dat = dat['questions']
   wordsX = []
   wordsY = []

   for e in dat:
      wordsX += e['question'].lower()[:-1].split()
      if 'answer' in e.keys():
         wordsY += e['answer'].lower().split()

   return wordsX + ['?'], wordsY

def name(split):
   return 'data/questions/CLEVR_' + split + '_questions.json'

if __name__ == '__main__':
   trainX, trainY = getAllWords(name('train'))
   valX, valY     = getAllWords(name('val'))
   testX, testY   = getAllWords(name('test'))

   #Probably should exclude val/test
   X = trainX + valX + testX
   Y = trainY + valY + testY

   X = np.unique(X)
   Y = np.unique(Y)

   X = ' '.join(X)
   Y = ' '.join(Y)

   with open('XVocab.txt', 'w') as f:
      f.write(X)

   with open('YVocab.txt', 'w') as f:
      f.write(Y)
