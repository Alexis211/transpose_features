import csv
import numpy
import h5py
from fuel.datasets.hdf5 import H5PYDataset


def build_dataset():

    train = numpy.zeros((38, 7130))
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

    test = numpy.zeros((34, 7130))
    with open("AMLALL_test.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:

            # Replace the class by 0 or 1
            if (row[-1] == "AML"):
                row[-1] = 0
            else:
                row[-1] = 1
            test[i, :] = row
            i += 1

    # Suffle the rows
    numpy.random.shuffle(train)
    numpy.random.shuffle(test)

    features = train[:, :-1].T
    big = numpy.concatenate((train, test), axis=0)

    # Convert each sample x_i into p samples (p is the number of features)
    # where each new sample = feature_j concat x_i_j
    new_dataset = [
        numpy.zeros((7129 * (38 + 34), 39)), numpy.zeros((7129 * (38 + 34)))]

    for i in range(7129):
        for j in range(38 + 34):
            new_sample = numpy.concatenate(
                (features[i, :], big[j, i:i + 1]), axis=1)
            new_target = big[j, -1]
            new_dataset[0][i * (38 + 34) + j, :] = new_sample
            new_dataset[1][i * (38 + 34) + j] = new_target

    f = h5py.File('leukemia_data.hdf5', mode='w')
    x = f.create_dataset('features', (7129 * (38 + 34), 39), dtype="float32")
    target = f.create_dataset('targets', (7129 * (38 + 34),), dtype='int32')

    x[...] = new_dataset[0].astype(numpy.float32)
    target[...] = new_dataset[1].astype(numpy.int32)

    split_dict = {
        'train': {'features': (0, 7129 * 38), 'targets': (0, 7129 * 38)},
        'test': {'features': (7129 * 38, 7129 * (38 + 34)), 'targets': (7129 * 38, 7129 * (38 + 34))}
    }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

if __name__ == "__main__":
    build_dataset()
