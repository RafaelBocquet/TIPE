#!/usr/bin/env python

import math
from decimal import Decimal

print "\n---- exp table 1 ----\n"
for i in range(1, 32):
  print i, ", ", 1 << i, ", ", Decimal(math.log(1 << i))
print "\n---- exp table 2 ----\n"
for i in range(1, 32):
  print Decimal(math.log(1.0 * (2 ** (-i)) + 1.0))
print "\n---- exp table 3 ----\n"
for i in range(1, 32):
  print Decimal((2**i * 1.0) / (1 + 2**i))