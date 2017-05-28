from pdb import set_trace as T
import json
import os
import numpy as np
import time
import h5py

from scipy.ndimage import imread
from scipy.ndimage import zoom

from lib import nlp
from Tree import BTree

def preprocessQuestions(questions, vocab, fAppend, maxLen=45):
   ret = []
   retMask = []
   retImgIdx= []
   for e in questions:
      retImgIdx += [e['image_index']]
      e = (e['question'].lower()[:-1] + ' ?').split()
      x = nlp.applyVocab(e, vocab).tolist()
      retMask += [len(x)]
      x += [0]*(maxLen - len(x))
      ret += [x]

   ret = np.asarray(ret)
   retMask = np.asarray(retMask)
   retImgIdx = np.asarray(retImgIdx)
   with h5py.File('data/preprocessed/clevr.h5', 'a') as f:
      data = f.create_dataset(fAppend+'Questions', data=ret)
      data = f.create_dataset(fAppend+'QuestionMask', data=retMask)
      data = f.create_dataset(fAppend+'ImageIdx', data=retImgIdx)

def preprocessAnswers(answers, vocab, fAppend):
   ret = []
   for e in answers:
      e = e['answer'].lower().split()
      x = nlp.applyVocab(e, vocab).tolist()
      ret += [x]

   ret = np.asarray(ret)
   with h5py.File('data/preprocessed/clevr.h5', 'a') as f:
      data = f.create_dataset(fAppend+'Answers', data=ret)

def preprocessPrograms(programs, vocab, fAppend, maxLen=45):
   ret = []
   retMask = []
   for p in programs:
      p = p['program']
      p = BTree(p).flat()
      p = nlp.applyVocab(p, vocab).tolist()
      retMask += [len(p)]
      p = [p + (45-len(p))*[0]]
      ret += p

   ret = np.asarray(ret).astype(np.int)
   retMask = np.asarray(retMask)
   with h5py.File('data/preprocessed/clevr.h5', 'a') as f:
      data = f.create_dataset(fAppend+'Programs', data=ret)
      data = f.create_dataset(fAppend+'ProgramMask', data=retMask)

def runTxt():
   splits = ['train', 'val']
   for split in splits:
      with open('data/clevr/questions/CLEVR_'+split+'_questions.json') as f:
         split = split[0].upper() + split[1:] 
         dat = json.load(f)['questions']

         print('Preprocessing Questions...')
         questionF = 'data/vocab/QuestionVocab.txt'
         questionVocab, _ = nlp.buildVocab(questionF, word=True)
         preprocessQuestions(dat, questionVocab, split)

         print('Preprocessing Answers...')
         answerF = 'data/vocab/AnswerVocab.txt'
         answerVocab, _ = nlp.buildVocab(answerF, word=True)
         preprocessAnswers(dat, answerVocab, split)

         print('Preprocessing Programs...')
         programF = 'data/vocab/ProgramVocab.txt'
         programVocab, _ = nlp.buildVocab(programF, word=True)
         preprocessPrograms(dat, programVocab, split)

         print('Done')

def runImgs():
   print('Preprocessing Images. This might take a while...')
   splits = ['train', 'val']
   for split in splits:
      imgNames = sorted(os.listdir('data/clevr/images/'+split))
      imgInds  = [int(e[12:-4]) for e in imgNames]
      numImgs = len(imgInds)

      imgMean = np.array([0.485, 0.456, 0.406])[None, None, None, :]
      imgStd  = np.array([0.229, 0.224, 0.225])[None, None, None, :]

      fAppend = split[0].upper() + split[1:]
      with h5py.File('data/preprocessed/clevr.h5', 'a') as f:
         data = f.create_dataset(fAppend+'Imgs', (numImgs, 224, 224, 3))

         imgs = []
         ind = 0
         for name in imgNames: 
            imgs += [imread('data/clevr/images/'+split+'/'+name)]

            ind += 1
            if ind % 1000 == 0:
               print(ind)
               imgs = np.stack(imgs).astype(np.float32)[:, :, :, :-1]
               imgs = imgs[:, 48:-48, 128:-128, :]/255.0
               imgs = (imgs - imgMean) / imgStd
               data[ind-1000:ind] = imgs
               imgs = []

   
