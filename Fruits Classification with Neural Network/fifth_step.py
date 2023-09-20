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

class NeuralNetworkFifthStep:
    def __init__(self, x_set, y_set):
        self.x_set = x_set
        self.y_set = y_set
        self.learning_rate = 1
        self.epoch_nums = 10
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

        tmp_for_layer3 = 2 * (self.a3 - y) * self.activation_derriative(self.z3)
        self.grad_w2 += tmp_for_layer3 @ (np.transpose(self.a2))
        self.grad_b2 += tmp_for_layer3

        # Calculating dcost/da2
        grad_a2 = (np.transpose(self.w2)) @ (2 * (self.a3 - y) * self.activation_derriative(self.z3))

        tmp_for_layer_2 = grad_a2 * self.activation_derriative(self.z2)
        self.grad_w1 += tmp_for_layer_2 @ (np.transpose(self.a1))
        self.grad_b1 += tmp_for_layer_2

        # Calculating dcost/da1
        grad_a1 = (np.transpose(self.w1)) @ (grad_a2 * self.activation_derriative(self.z2))

        tmp_for_layer1 = grad_a1 * self.activation_derriative(self.z1)
        self.grad_w0 += tmp_for_layer1 @ (np.transpose(x))
        self.grad_b0 += tmp_for_layer1

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
        return x, y

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
                # print(f"Batch {batch + 1} finished!")
            print(f"Epoch {i + 1} finished!")
            self.cost_function()
            self.cost_array.append(self.cost)
        end_time = time.time()
        print(f"It took {end_time - start_time} seconds !")
        plt.plot(self.cost_array)
        plt.show()
        acc=self.accuracy()
        return acc

    def accuracy(self):
        correct_predictions = 0
        for i in range(0, len(self.x_set)):
            x = self.x_set[i]
            y = self.y_set[i]
            self.feed_forward(x)
            if np.argmax(self.a3) == np.argmax(y):
                correct_predictions += 1
        print("Accuracy for Train in fourth step is : " + str(1.0 * correct_predictions / len(self.x_set)))
        return 1.0*correct_predictions/len(self.x_set)

    def accuracy_for_test(self,x_test,y_test):
        correct_predictions = 0
        for i in range(0, len(x_test)):
            x = x_test[i]
            y = y_test[i]
            self.feed_forward(x)
            if np.argmax(self.a3) == np.argmax(y):
                correct_predictions += 1
        print("Accuracy for Test in fourth step is : " + str(1.0 * correct_predictions / len(x_test)))
        return 1.0*correct_predictions/len(x_test)

train_set_for_fifth_step = train_set
test_set_for_fifth_step=test_set
x_set_for_fifth_step = []
y_set_for_fifth_step = []
x_test=[]
y_test=[]
for i in range(0, len(train_set_for_fifth_step)):
    x_set_for_fifth_step.append(train_set_for_fifth_step[i][0])
    y_set_for_fifth_step.append(train_set_for_fifth_step[i][1])
for i in range(0,len(test_set_for_fifth_step)):
    x_test.append(test_set_for_fifth_step[i][0])
    y_test.append(test_set_for_fifth_step[i][1])

average_train_accuracy=0
average_test_accuracy=0
for i in range(0,10):
    neural_network_fifthStep = NeuralNetworkFifthStep(x_set_for_fifth_step, y_set_for_fifth_step)
    average_train_accuracy+=neural_network_fifthStep.train_data()
    average_test_accuracy+=neural_network_fifthStep.accuracy_for_test(x_test,y_test)
average_train_accuracy/=10
average_test_accuracy/=10
print("*******************************************************")
print(f"Average Train accuracy is : {average_train_accuracy*100} %")
print(f"Average Test accuracy is : {average_test_accuracy*100} %")