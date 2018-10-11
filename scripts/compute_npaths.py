import sys
import numpy as np

data = open(sys.argv[1]).readlines()

print("Npath total: %s" % sum([int(x) for x in data]))
print("Npath max: %s" % max([int(x) for x in data]))
print("Npath mean: %s" % np.mean([int(x) for x in data]))
print("Npath funcs: %s" % len(data))
