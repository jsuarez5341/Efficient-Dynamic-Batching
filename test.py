from clevrBatcher import ClevrBatcher
from pdb import set_trace as t

if __name__ == '__main__':
   batcher = ClevrBatcher('train')
   dat = batcher.next()
   t()
