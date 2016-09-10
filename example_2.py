import csv
from sklearn import svm, metrics
from numpy import genfromtxt
import numpy as np

#open the file and cut the first line
dataset = genfromtxt('train.csv', dtype=np.dtype('>i4'), delimiter=',')
datadset = dataset[1:]

labels = [x[0] for x in dataset]
data = [x[1:] for x in dataset]

number_of_features = len(data)
number_of_samples = len(labels)

print("Number of samples: " + str(number_of_samples) + \
      ", number of features: " + str(number_of_features))

#2/3 of the data to learning, 1/3 of data to testing
split_point = int(number_of_samples * 0.66)

labels_learn = labels[:split_point]
data_learn = data[:split_point]

labels_test = labels[split_point:]
data_test = data[split_point:]

print("Training: " + str(len(labels_learn)) + " Test: " + str(len(labels_test)))

classifier = svm.SVC(kernel='poly', degree=2, C=1.0)
classifier.fit(data_learn, labels_learn)

predicted = classifier.predict(data_test)

print("Classification report for classifier %s:\n%s\n" % \
      (classifier, metrics.classification_report(labels_test, predicted)))

print("Confusion matrix:\n%s" % \
      metrics.confusion_matrix(labels_test, predicted))
