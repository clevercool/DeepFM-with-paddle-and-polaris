import pandas as pd
import numpy as np

data1 = pd.read_csv("/mnt/data/test.set.a.txt", header=None)
data2 = pd.read_csv("p1.txt", header=None)
pre = np.array(data2[0].astype(float))
res = np.array(data1[1].astype(int))

n = 0
for i in range(len(pre)):
 tem = 1
 if(pre[i] < 0.5):
  tem = 0
 if(res[i] == tem):
  n+=1

print(n)
r = float(n)/1000000
print(r)


