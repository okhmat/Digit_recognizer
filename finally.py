import numpy as np
import numpy.testing
import unittest
import csv
from sklearn import svm
import matplotlib.pyplot as plt


def train_data(num_of_digits, file_path):
    labels = np.zeros([42000, 1], dtype=np.int32)
    pictures = np.zeros([42000, 28, 28], dtype=np.int32)

    in_file = csv.reader(open(file_path))
    next(in_file)

    #fit the labels and pictures arrays
    i = 0
    j = 0
    for row in in_file:
        for j in range(0, 784):
            row[j] = int(row[j])
        labels[i, 0] = row[0]
        row = np.reshape(row[1:], (28, 28))
        pictures[i, ] = np.asarray(row)
        i = i+1
        if i == num_of_digits:
            break
    pictures = np.reshape(pictures, (42000, 784))
    return pictures, labels

print(train_data(10, 'train.csv'))

def test_data():
    file = csv.reader(open('test.csv'))
    #cut the first line
    next(file)
    print("the result of function next", next(file))
    testing_data = []
    for row in file:
        aux_arr = np.asarray(row, dtype=np.int32)
        testing_data.append(aux_arr)
    return testing_data

#it makes a .csv file of predicted data (concatination pictures with predicted
#labels)
def csv_maker(sample):
    sample = np.asarray(sample, dtype=np.str)
    writer = csv.writer(open('predicted.csv', 'w'), quoting=csv.QUOTE_NONNUMERIC, \
                        lineterminator='\n')
    writer.writerow(['ImageId', "Label"])
    for i in range(0, len(sample)):
        writer.writerow([i + 1, sample[i]])

num_of_digits = 100
labels, pictures = train_data(num_of_digits, 'train.csv')
print(pictures.shape, labels.shape)
npa = np.ndarray([])

# learning
classifier = svm.SVC(kernel='linear', C=1.0).fit(pictures, labels)
# classifier = svm.SVC(kernel='poly', degree=3, C=1.0)
# classifier.fit()
csv_maker(classifier.predict(np.c_[test_data()]))

# numpy.testing.assert_array_almost_equal(, reader_test())

if __name__ == '__main__':
    unittest.main()