import csv
import numpy
import h5py
from fuel.datasets.hdf5 import H5PYDataset



def build_dataset():
    train_ex = 38
    test_ex = 34
    feat = 7129

    train = numpy.zeros((train_ex, feat + 1))
    with open("AMLALL_train.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:

            # Replace the class by 0 or 1
            if (row[-1] == "AML"):
                row[-1] = 0
            else:
                row[-1] = 1
            train[i, :] = row
            i += 1


    # Suffle the examples
    numpy.random.shuffle(train)

    # features has shape = (feat, train_ex)
    features = train[:, :-1].T
    # targets has shape = (feat, train_ex)
    targets = numpy.dot(numpy.ones((feat,1)),train[:,-1].reshape((1, train_ex)))

    f = h5py.File('leukemia_transpose.hdf5', mode='w')
    x = f.create_dataset('features', (feat, train_ex), dtype="float32")
    target = f.create_dataset('targets', (feat, train_ex), dtype='int32')

    x[...] = features.astype(numpy.float32)
    target[...] = targets.astype(numpy.int32)

    split_dict = {
        'train': {'features': (0, feat), 'targets': (0, feat)},
    }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

if __name__ == "__main__":
    build_dataset()
