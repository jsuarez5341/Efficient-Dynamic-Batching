from pdb import set_trace as T
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import utils

class ProgramGenerator(nn.Module):
   def __init__(self, 
         embedDim, hGen, qLen, qVocab, pVocab):
      super(ProgramGenerator, self).__init__()
      self.embed = nn.Embedding(qVocab, embedDim)
      self.encoder = t.nn.LSTM(embedDim, hGen, 1, batch_first=True)
      self.decoder = t.nn.LSTM(hGen, hGen, 1, batch_first=True)
      self.proj  = nn.Linear(hGen, pVocab)

      self.qLen = qLen
      self.hGen = hGen
      self.pVocab = pVocab

      #For REINFORCE
      self.eda = utils.EDA()

   def forward(self, x, trainable=False):
      x = self.embed(x)
      x, state = self.encoder(x)
      state = [state[0][0] for i in range(self.qLen)]
      state = t.stack(state, 1)
      x, _ = self.decoder(x)
      x = x.contiguous().view(-1, self.hGen)
      x = self.proj(x)

      #x = F.softmax(x)
      #reward = x.data
      #self.eda.update(reward.cpu().numpy())

      #if trainable:
      #   x = x.multinomial(pVocab)
      #   x.reinforce(reward)

      #x = x.view(-1, self.qLen, self.pVocab)
      #_, x = t.max(x, 2) #Program inds
      #x = t.squeeze(x)
      return x
