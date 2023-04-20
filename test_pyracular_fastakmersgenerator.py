import pyracular
import numpy as np
import time

k = 21
batch_size = 128
window_size = 16

fasta = pyracular.FastaKmersGenerator(k, "/mnt/ramfs/nt_train.sfasta", window_size, True, False)

start_time = time.process_time()

total = 0
for i in fasta:
  total = total + 1
#  print(total)
  if total % 100000 == 0:
    end_time = time.process_time()
    print("10k in {}".format(end_time - start_time))
    start_time = time.process_time()
#    print(total)
#  print(len(i['kmers'][0]))
#  if total == 100000:
#    break

#end_time = time.time()

#time_taken = end_time - start_time
#print(time_taken / 10000)
print(total)

