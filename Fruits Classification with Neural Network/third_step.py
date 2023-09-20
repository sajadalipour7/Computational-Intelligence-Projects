import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import time


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


class NeuralNetworkThirdStep:
    def __init__(self, x_set, y_set):
        self.x_set = x_set
        self.y_set = y_set
        self.learning_rate = 1
        self.epoch_nums = 5
        self.batch_size = 10
        self.w0 = np.random.normal(0, 1, size=(150, 102))
        self.w1 = np.random.normal(0, 1, size=(60, 150))
        self.w2 = np.random.normal(0, 1, size=(4, 60))
        self.grad_w0 = np.zeros((150, 102))
        self.grad_w1 = np.zeros((60, 150))
        self.grad_w2 = np.zeros((4, 60))
        self.b0 = np.zeros((150, 1))
        self.b1 = np.zeros((60, 1))
        self.b2 = np.zeros((4, 1))
        self.grad_b0 = np.zeros((150, 1))
        self.grad_b1 = np.zeros((60, 1))
        self.grad_b2 = np.zeros((4, 1))
        self.z1 = None
        self.z2 = None
        self.z3 = None
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.cost = 0
        self.cost_array = []

    def activation(self, input):
        return 1 / (1 + np.exp(-input))

    def activation_derriative(self, input):
        s = 1 / (1 + np.exp(-input))
        return np.multiply(s, (1 - s))

    def feed_forward(self, x):
        self.z1 = np.dot(self.w0, x) + self.b0
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.w1, self.a1) + self.b1
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.w2, self.a2) + self.b2
        self.a3 = self.activation(self.z3)

    def backpropagation(self, x, y):
        self.feed_forward(x)
        for i in range(0, self.w2.shape[0]):
            for j in range(0, self.w2.shape[1]):
                tmp = 2 * (self.a3[i] - y[i]) * self.activation_derriative(self.z3[i])
                self.grad_w2[i][j] += tmp * self.a2[j]
            self.grad_b2[i] += 2 * (self.a3[i] - y[i]) * self.activation_derriative(self.z3[i])

        # Calculating dcost/da2
        grad_a2 = np.zeros(self.a2.shape)
        for i in range(0, self.a2.shape[0]):
            for j in range(0, self.a3.shape[0]):
                grad_a2[i] += 2 * (self.a3[j] - y[j]) * self.activation_derriative(self.z3[j]) * self.w2[j, i]

        for i in range(0, self.w1.shape[0]):
            for j in range(0, self.w1.shape[1]):
                tmp = grad_a2[i] * self.activation_derriative(self.z2[i])
                self.grad_w1[i][j] += tmp * self.a1[j]
            self.grad_b1[i] += grad_a2[i] * self.activation_derriative(self.z2[i])

        # Calculating dcost/da1
        grad_a1 = np.zeros(self.a1.shape)
        for i in range(0, self.a1.shape[0]):
            for j in range(0, self.a2.shape[1]):
                grad_a1[i] += grad_a2[j] * self.activation_derriative(self.z2[j]) * self.w1[j, i]

        for i in range(0, self.w0.shape[0]):
            for j in range(0, self.w0.shape[1]):
                tmp = grad_a1[i] * self.activation_derriative(self.z1[i])
                self.grad_w0[i][j] += tmp * x[j]
            self.grad_b0 += grad_a1[i] * self.activation_derriative(self.z1[i])

    def cost_function(self):
        self.cost = 0
        for i in range(0, len(self.x_set)):
            x = self.x_set[i]
            y = self.y_set[i]
            self.feed_forward(x)
            error = self.a3 - y
            squared_error = np.square(error)
            self.cost += np.sum(squared_error)
        self.cost /= len(self.x_set)

    def shuffle(self):
        x = []
        y = []
        total = []
        for i in range(0, len(self.x_set)):
            total.append((self.x_set[i], self.y_set[i]))
        random.shuffle(total)
        for i in range(0, len(total)):
            x.append(total[i][0])
            y.append(total[i][1])
        return (x, y)

    def clean_gradients(self):
        self.grad_w0 = np.zeros((150, 102))
        self.grad_w1 = np.zeros((60, 150))
        self.grad_w2 = np.zeros((4, 60))
        self.grad_b0 = np.zeros((150, 1))
        self.grad_b1 = np.zeros((60, 1))
        self.grad_b2 = np.zeros((4, 1))

    def train_data(self):
        self.cost_function()
        self.cost_array.append(self.cost)
        start_time = time.time()
        for i in range(0, self.epoch_nums):
            shuffled_x, shuffled_y = self.shuffle()
            number_of_batches = len(self.x_set) / self.batch_size
            if number_of_batches != int(number_of_batches):
                number_of_batches = int(number_of_batches) + 1
            number_of_batches = int(number_of_batches)
            for batch in range(0, number_of_batches):
                X_set = shuffled_x[batch * self.batch_size:min(len(self.x_set), (batch + 1) * self.batch_size)]
                Y_set = shuffled_y[batch * self.batch_size:min(len(self.x_set), (batch + 1) * self.batch_size)]
                self.clean_gradients()
                for n in range(0, len(X_set)):
                    self.backpropagation(X_set[n], Y_set[n])
                self.w0 -= self.learning_rate * (1.0 * self.grad_w0 / len(X_set))
                self.w1 -= self.learning_rate * (1.0 * self.grad_w1 / len(X_set))
                self.w2 -= self.learning_rate * (1.0 * self.grad_w2 / len(X_set))
                self.b0 -= self.learning_rate * (1.0 * self.grad_b0 / len(X_set))
                self.b1 -= self.learning_rate * (1.0 * self.grad_b1 / len(X_set))
                self.b2 -= self.learning_rate * (1.0 * self.grad_b2 / len(X_set))
                print(f"Batch {batch + 1} finished!")
            print(f"Epoch {i + 1} finished!")
            self.cost_function()
            self.cost_array.append(self.cost)
        end_time = time.time()
        print(f"It took {end_time - start_time} seconds !")
        plt.plot(self.cost_array)
        plt.show()
        self.accuracy()

    def accuracy(self):
        correct_predictions = 0
        for i in range(0, len(self.x_set)):
            x = self.x_set[i]
            y = self.y_set[i]
            self.feed_forward(x)
            if np.argmax(self.a3) == np.argmax(y):
                correct_predictions += 1
        print("Accuracy for third step is : " + str(1.0 * correct_predictions / len(self.x_set)))


train_set_for_third_step = train_set[:200]
x_set_for_third_step = []
y_set_for_third_step = []
for i in range(0, len(train_set_for_third_step)):
    x_set_for_third_step.append(train_set_for_third_step[i][0])
    y_set_for_third_step.append(train_set_for_third_step[i][1])
neural_network_thirdStep = NeuralNetworkThirdStep(x_set_for_third_step, y_set_for_third_step)
neural_network_thirdStep.train_data()
