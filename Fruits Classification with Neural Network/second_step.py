import numpy as np
import random
import pickle
import matplotlib.pyplot as plt


def load_dataset():
    # loading training set features
    f = open("Datasets/train_set_features.pkl", "rb")
    train_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=train_set_features2, axis=0)
    train_set_features = train_set_features2[:, features_STDs > 52.3]

    # changing the range of data between 0 and 1
    train_set_features = np.divide(train_set_features, train_set_features.max())

    # loading training set labels
    f = open("Datasets/train_set_labels.pkl", "rb")
    train_set_labels = pickle.load(f)
    f.close()

    # ------------
    # loading test set features
    f = open("Datasets/test_set_features.pkl", "rb")
    test_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=test_set_features2, axis=0)
    test_set_features = test_set_features2[:, features_STDs > 48]

    # changing the range of data between 0 and 1
    test_set_features = np.divide(test_set_features, test_set_features.max())

    # loading test set labels
    f = open("Datasets/test_set_labels.pkl", "rb")
    test_set_labels = pickle.load(f)
    f.close()

    # ------------
    # preparing our training and test sets - joining datasets and lables
    train_set = []
    test_set = []

    for i in range(len(train_set_features)):
        label = np.array([0, 0, 0, 0])
        label[int(train_set_labels[i])] = 1
        label = label.reshape(4, 1)
        train_set.append((train_set_features[i].reshape(102, 1), label))

    for i in range(len(test_set_features)):
        label = np.array([0, 0, 0, 0])
        label[int(test_set_labels[i])] = 1
        label = label.reshape(4, 1)
        test_set.append((test_set_features[i].reshape(102, 1), label))

    # shuffle
    random.shuffle(train_set)
    random.shuffle(test_set)

    return (train_set, test_set)


train_set, test_set = load_dataset()


class NeuralNetworkSecondStep:
    def __init__(self, x_set, y_set):
        self.x_set = x_set
        self.y_set = y_set
        self.w0 = np.random.normal(size=(150, 102))
        self.w1 = np.random.normal(size=(60, 150))
        self.w2 = np.random.normal(size=(4, 60))
        self.b0 = np.zeros((150, 1))
        self.b1 = np.zeros((60, 1))
        self.b2 = np.zeros((4, 1))
        self.z1 = None
        self.z2 = None
        self.z3 = None
        self.a1 = None
        self.a2 = None
        self.a3 = None

    def activation(self, input):
        return 1.0 / (1 + np.exp(-1 * input))

    def feed_forward(self, x):
        self.z1 = np.dot(self.w0, x) + self.b0
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.w1, self.a1) + self.b1
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.w2, self.a2) + self.b2
        self.a3 = self.activation(self.z3)

    def accuracy(self):
        correct_predictions = 0
        for i in range(0, len(self.x_set)):
            x = self.x_set[i]
            y = self.y_set[i]
            self.feed_forward(x)
            if np.argmax(self.a3) == np.argmax(y):
                correct_predictions += 1
        print("Accuracy for second step is : " + str(1.0 * correct_predictions / len(self.x_set)))


train_set_for_second_step = train_set[:200]
x_set_for_second_step = []
y_set_for_second_step = []
for i in range(0, len(train_set_for_second_step)):
    x_set_for_second_step.append(train_set_for_second_step[i][0])
    y_set_for_second_step.append(train_set_for_second_step[i][1])
neural_network_secondStep = NeuralNetworkSecondStep(x_set_for_second_step, y_set_for_second_step)
neural_network_secondStep.accuracy()
