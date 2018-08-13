import pandas as pd


def pre(data):
  for i in range(1000000):
    if(data[0][i] > 0.5):
      data[0][i] = 1
    else:
      data[0][i] = 0

data = pd.read_csv("./predict.txt", header=None)

pre(data)
data

