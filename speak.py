import os
import numpy as np
import sys
import my_pb_mod

print("python is", sys.version_info)

print("imported, about to call", file=sys.stderr)
result = my_pb_mod.add(2, 3)
