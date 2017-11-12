#!/usr/bin/env python3

import sys
import random

in_path = sys.argv[1]
out_path = in_path[:in_path.index('.')] + '.trim' + in_path[in_path.index('.'):]

read = open(in_path, 'r')
write = open(out_path, 'w')

data = []

for line in read:
    line = line.split(',')
    data.append(line)

MAX_POINTS = 1000
count = 0
while count < MAX_POINTS:
    point = random.randint(0, len(data) - 1)
    write.write(','.join(data[point]))
    del data[point]
    count += 1

read.close()
write.close()
