import numpy as np
import csv
from sklearn import svm

inFile = open('train.csv', 'r')
i = 0
string = []  #row is the same with string number
for row in csv.reader(inFile, delimiter = ','):
    string[i] = row
    i = i+1
    #считываем 100 строчек

    if i == 100:
        break
#открыть как массив строчек а потом работать с каждой строчкой
print(string)
j = 0
i = 1
pixelArray = np.array([])
column = []
mega = []
mega1 = np.array([])
for row in csv.reader(inFile, delimiter = ','):
    column = row
    column = column[1:]
    for j in range(0, 28*28):
        column[j] = int(column[j])
        #add a value to the end of array
    pixelArray = np.append(pixelArray, column)
    mega = int(row[0])
    mega1 = np.append(mega1, mega)
    i = i+1
    if i == 1000:
        break
inFile.close()
print(mega1)
column = column.reshape(1000, 784)
print(column)

C = 1.0
#далее обучаемся
svc = svm.LinearSVC(C = C).fit(column, mega1)
print(svc)

inFile = open('test.csv', 'r')
#for row in csv.reader(inFile, delimiter = ','):
inFile.close()

