from main import *
from little_tools import print_obj

conll = D.CoNLLDataset(datadir, conll_path, person_path, args.order, args.method)
print_obj(conll, 'conll', depth=0, max_bro=14)
