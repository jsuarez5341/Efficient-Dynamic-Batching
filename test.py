from ClevrBatcher import ClevrBatcher
from pdb import set_trace as T

if __name__ == '__main__':
   batcher = ClevrBatcher(64, 'Train')
   dat = batcher.next()
   T()
   pass
