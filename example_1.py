import sklearn
import matplotlib.pyplot as plt
from sklearn import svm, metrics, datasets
import numpy as np
import csv

digits = open('train.csv', 'r')

i = 0
a = []

# Считываем из файла первую строчку, в которой наименования столбцов; они нам
# не нужны.
for row in csv.reader(digits, delimiter=','):
    a = row
    i = i+1
    if i == 1:
        break

i = 0
j = 0

# Делаем массив pictures из строчек; одна строчка - одна картинка;
# строчки размера 1*784 превратим в матрицы 28*28 и
# сделаем значения элементы этих матриц целыми числами; сейчас это символы.
# В массиве values - соответствующие картинкам значения написанных на них цифр.

pictures = np.ndarray((28, 28), dtype=int)
pictures = np.zeros();
values = np.ndarray((1, 42000), dtype=int)

# Продолжаем читать дальше; считаем следующие 10 строчек
for row in csv.reader(digits, delimiter=','):

    # Сделаем все элементы считанной строчки целыми числами
    for j in range(0, 784):
        row[j] = int(row[j])

    # Теперь эту строчку из целых чисел нужно поделить на значение нарисованной
    # цифры и на "пиксели"
    values = np.append(values, row[0])
    #pictures = np.append(pictures, row[1:])
    i = i+1
    if i == 1:
        break

# Обучимся на загруженной выборке
values = values[2:]

print(values)
print(pictures)

#svc = svm.SVC(kernel = 'linear', C = 1.0)
#svc = svc.fit(pictures, values)