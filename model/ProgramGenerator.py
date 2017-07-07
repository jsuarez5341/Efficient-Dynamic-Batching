from pdb import set_trace as T
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from lib import utils

class ProgramGenerator(nn.Module):
   def __init__(self, 
         embedDim, hGen, qLen, qVocab, pVocab):
      super(ProgramGenerator, self).__init__()
      self.embed = nn.Embedding(qVocab, embedDim)
      self.encoder = t.nn.LSTM(embedDim, hGen, 2, batch_first=True)
      self.decoder = t.nn.LSTM(hGen, hGen, 2, batch_first=True)
      self.proj  = nn.Linear(hGen, pVocab)

      self.qLen = qLen
      self.hGen = hGen
      self.pVocab = pVocab

   def forward(self, x):
      x = self.embed(x)
      x, state = self.encoder(x)
      stateInp = [state[0][0] for i in range(self.qLen)]
      stateInp = t.stack(stateInp, 1)
      x, _ = self.decoder(stateInp, state)

      sz = list(x.size())
      x  = x.contiguous().view(-1, self.hGen)
      x  = self.proj(x)
      x  = x.view(*sz[:2], -1)

      return x
