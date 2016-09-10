import numpy as np
import csv
from sklearn import svm
import matplotlib.pyplot as plt

i = 0
a = []
b = []
c = []
k = []

npb = np.array([])
npc = np.array([])
npk = np.array([])

inFile = open('train.csv', 'r')

#read array of pixels line by line into array a[]
for row in  csv.reader(inFile, delimiter = ','):
    a = row
    i = i+1
    if i == 1:
        break
#тут строчка с именами пикселей(нам похуй на нее)

i = 0

for row in csv.reader(inFile, delimiter = ','):
    b = row
    b = b[1:]   #cut the first element in line which is a target for prediction
    for j in range(0, 28*28):
        b[j] = int(b[j])    #lead array of symbols to array of integers
    npb = np.append(npb, b)
    c = int(row[0])
    npc = np.append(npc, c)
    i = i+1
    if i == 1000:
        break

inFile.close()
print(npc)
npb = npb.reshape(10, 784)
print(npb)

#learning
C = 1.0
svc = svm.SVC(kernel = 'linear', C = C).fit(npb, npc)

#now we will test our classifer

inFile = open('test.csv', 'r')

i = 0
for row in csv.reader(inFile, delimiter = ','):
    i = i+1
    if i == 1000:
        break

i = 0
for row in csv.reader(inFile, delimiter = ','):
    k = row
    for j in range(0, 784):
        k[j] = int(k[j])
    npk = np.append(npk, k)
    i = i+1
    if i == 10:
        break
inFile.close()
npk = npk.reshape(10, 784)
print(npk)