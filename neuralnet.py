import math

import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

class dataProcess():
    def load_data(self, filename):
        self.y_label = []
        self.x_input = []
        with open(filename, newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter=",")
            #print(reader)
            for row in reader:
                self.y_label.append(row[0])
                design_xi = np.array(row[1:])
                self.x_input.append(design_xi)
        self.y_label = np.array(self.y_label, dtype = float)
        self.y_label = np.rint(self.y_label)
        self.y_label = self.y_label.astype(int)
        self.x_input = np.array(self.x_input, dtype = float)
    def x_process(self):
        bias_x = np.ones(len(self.x_input))
        self.x_neural = np.insert(self.x_input, 0, bias_x, axis = 1)
    def y_onehot_code(self):
        one_hot = np.zeros([len(self.y_label), 4])
        rows = np.arange(self.y_label.size)
        one_hot[rows, self.y_label] = 1
        self.y_onehot = one_hot


class neural_nn():
    # input bias x_input[i] column vector
    #X_input = sample rows
    #y_label = columm vector y one hot
    def __init__(self, hidden_unit_num, x_input, k_class, y_label,learning_rate, init_flag):
        self.hidden_unit_num = hidden_unit_num
        self.x_input = x_input
        self.input_len = len(x_input[0])
        self.k = k_class
        #self.y_label = np.array([y_label]).T
        self.y_label = y_label
        self.learning_rate = learning_rate
        self.init_flag = init_flag
    def init_parameter(self):
        #alpha size = hidden layer * x
        #beta_size = output layer (k) * hidden layer+1
        if(self.init_flag == 1):
            alpha = np.random.uniform(-0.1,0.1,size=(self.hidden_unit_num, self.input_len-1))
            beta =  np.random.uniform(-0.1,0.1,size=(self.k, self.hidden_unit_num))
        elif(self.init_flag == 2):
            alpha = np.zeros([self.hidden_unit_num, self.input_len-1])
            beta = np.zeros([self.k, self.hidden_unit_num])
        #append bias weight
        alpha_bias = np.zeros(self.hidden_unit_num)
        self.alpha = np.insert(alpha, 0, alpha_bias, axis = 1)
        beta_bias = np.zeros(self.k)
        self.beta = np.insert(beta, 0, beta_bias, axis = 1)
        #initial st alpha
        self.st_alpha = np.zeros([self.hidden_unit_num, self.input_len])
        self.st_beta = np.zeros([self.k, self.hidden_unit_num+1])

    # calculate matrix sigmoid
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def softmax(self, x):
        exp_x = np.exp(x)
        y = exp_x / np.sum(exp_x)
        return y

    def cal_loss(self, y_hat, y):
        logy_hat = np.log(y_hat)
        return (-np.dot(y.T, logy_hat))

    #i sample forward
    def nn_forward(self, i):
        #hidden_a = hidden layer * 1
        x_i_input = np.array([self.x_input[i]]).T
        y_i_label = np.array([self.y_label[i]]).T
        self.hidden_a = np.dot(self.alpha, x_i_input)
        self.hidden_z = self.sigmoid(self.hidden_a)
        #hidden_z = hidden_layer+1 * 1
        self.hidden_z = np.insert(self.hidden_z, 0, np.ones(1), axis = 0)
        # b = k class columns
        self.b = np.dot(self.beta, self.hidden_z)
        self.y_hat = self.softmax(self.b)
        self.cross_entropy = self.cal_loss(self.y_hat, y_i_label)

    def nn_backward(self, i):
        x_i_input = np.array([self.x_input[i]]).T
        y_i_label = np.array([self.y_label[i]]).T
        # row vector 1 * k_class
        self.partial_loss_b = np.transpose(self.y_hat - y_i_label)
        # matrix k_class * hidden unit+1
        self.partial_loss_beta = np.dot(np.transpose(self.partial_loss_b), np.transpose(self.hidden_z))
        #column vector hidden layer
        self.partial_loss_hidden_z =np.dot(self.partial_loss_b, self.beta[:,1:]).T
        # column vector hidden layer
        self.partial_loss_hidden_a = self.partial_loss_hidden_z * self.hidden_z[1:,:] * (1-self.hidden_z[1:,:] )
        #hidden layer * x
        self.partial_loss_alpha = np.dot(self.partial_loss_hidden_a, x_i_input.T)

    def update_parameter(self):
        self.st_alpha = self.st_alpha+self.partial_loss_alpha*self.partial_loss_alpha
        self.alpha = self.alpha - (self.learning_rate/np.sqrt(self.st_alpha + 1e-5))*self.partial_loss_alpha
        #print(np.sqrt(self.partial_loss_beta * self.partial_loss_beta))
        self.st_beta = self.st_beta + self.partial_loss_beta * self.partial_loss_beta
        self.beta = self.beta - (self.learning_rate / np.sqrt(self.st_beta + 1e-5)) * self.partial_loss_beta
        #self.alpha = self.alpha - self.learning_rate*self.partial_loss_alpha
        #self.beta = self.beta - self.learning_rate*self.partial_loss_beta

class train():
    def __init__(self, init_flag, epoch, learning_rate, x_input, y_input, hidden_layer, x_validaition, y_validation):
        self.init_flag = init_flag
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.x_input = x_input
        self.y_input = y_input
        self.hidden_layer = hidden_layer
        self.x_validation = x_validaition
        self.y_validation = y_validation
        self.train_entropy_error = []
        self.valid_entropy_error = []

    def train(self):
        self.neural_network = neural_nn(self.hidden_layer, self.x_input, 4, self.y_input, self.learning_rate, self.init_flag)
        self.neural_network.init_parameter()
        for i in range(self.epoch):
            #print(type(self.train_entropy_error))
            print(i)
            #print(self.neural_network.alpha)
            #print(self.neural_network.beta)
            train_entropy = 0
            for j in range(len(self.x_input)):
                self.neural_network.nn_forward(j)
                train_entropy = train_entropy+self.neural_network.cross_entropy
                self.neural_network.nn_backward(j)
                self.neural_network.update_parameter()
                #print(self.neural_network.alpha)
                #print(self.neural_network.beta)
             #evaluate cross entropy
            self.train_entropy_error.append(self.entropy_loss(self.x_input, self.y_input)[0][0])
            self.valid_entropy_error.append(self.entropy_loss(self.x_validation, self.y_validation)[0][0])
        #print(self.train_entropy_error)
        self.train_error_rate, self.trainout_label = self.cal_error_rate(self.x_input, self.y_input)
        self.valid_error_rate, self.validout_label = self.cal_error_rate(self.x_validation, self.y_validation)

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def softmax(self, x):
        exp_x = np.exp(x)
        y = exp_x / np.sum(exp_x)
        return y

    def cal_loss(self, y_hat, y):
        logy_hat = np.log(y_hat)
        return (-np.dot(y.T, logy_hat))
        #return (-y.T * logy_hat)

    #i sample forward
    def entropy_loss(self, x_validation, y_validation):
        #hidden_a = hidden layer * 1
        cross_entropy = 0
        error = 0
        for i in range(len(x_validation)):
            x_i_input = np.array([x_validation[i]]).T
            y_i_label = np.array([y_validation[i]]).T
            hidden_a = np.dot(self.neural_network.alpha, x_i_input)
            hidden_z = self.sigmoid(hidden_a)
            # hidden_z = hidden_layer+1 * 1
            hidden_z = np.insert(hidden_z, 0, np.ones(1), axis=0)
            # b = k class columns
            b = np.dot(self.neural_network.beta, hidden_z)
            y_hat = self.softmax(b)
            cross_entropy = cross_entropy+self.cal_loss(y_hat, y_i_label)
        return cross_entropy/len(x_validation)

    def cal_error_rate(self, x_input, y_input):
        error = 0
        predict_label = []
        for i in range(len(x_input)):
            x_i_input = np.array([x_input[i]]).T
            y_i_label = np.array([y_input[i]]).T
            hidden_a = np.dot(self.neural_network.alpha, x_i_input)
            hidden_z = self.sigmoid(hidden_a)
            # hidden_z = hidden_layer+1 * 1
            hidden_z = np.insert(hidden_z, 0, np.ones(1), axis=0)
            # b = k class columns
            b = np.dot(self.neural_network.beta, hidden_z)
            y_hat = self.softmax(b)
            y_predict = np.argmax(y_hat)
            #print(y_predict)
            predict_label.append(y_predict)
            if (y_i_label[y_predict] != 1):
                error = error + 1
        return error/(len(x_input)), predict_label



if __name__ == "__main__":
    '''
    traininput = sys.argv[1]
    validationinput = sys.argv[2]
    trainout = sys.argv[3]
    validout = sys.argv[4]
    metricout = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units =int (sys.argv[7])
    init_flag = int (sys.argv[8])
    learning_rate = float (sys.argv[9])
    '''
    traininput = "small_train.csv"
    validationinput = "small_val.csv"
    num_epoch = 100
    hidden_units = 50
    init_flag = 1
    learning_rate = 0.01

    dataProcess = dataProcess()
    dataProcess.load_data(traininput)
    dataProcess.x_process()
    train_x_input = dataProcess.x_neural
    #print(train_x_input)
    dataProcess.y_onehot_code()
    train_y_onehot = dataProcess.y_onehot
    #print(train_y_onehot)


    dataProcess.load_data(validationinput)
    dataProcess.x_process()
    dataProcess.y_onehot_code()
    x_valid = dataProcess.x_neural
    y_valid = dataProcess.y_onehot
    #print(x_valid)
    #print(y_valid)
    #3.1 a

    hidden_list = [5, 20, 50, 100,200]
    train_cross_entropy = []
    valid_cross_entropy = []
    for item in hidden_list:
        train_model = train(init_flag, num_epoch, learning_rate, train_x_input, train_y_onehot, int(item), x_valid, y_valid)
        train_model.train()
        train_entropy_error = train_model.train_entropy_error[-1]
        valid_entropy_error = train_model.valid_entropy_error[-1]
        train_cross_entropy.append(train_entropy_error)
        valid_cross_entropy.append(valid_entropy_error)

    plt.plot(hidden_list, train_cross_entropy, label="Average training cross-entropy error")
    plt.plot(hidden_list, valid_cross_entropy, label="Average validation cross-entropy error")
    # plt.plot(x_axis, likelihood3, label="alpha = 0.1")
    plt.xlabel("hidden units")  # x label
    plt.ylabel("Cross-entropy error")  # y label
    plt.legend()
    plt.show()

    '''
    #3.1 c
    sgd_error = []
    with open("val_loss_sgd_out.txt", newline='') as tsvfile:
        reader = csv.reader(tsvfile)
        # print(reader)
        for row in reader:
            #print(row[0])
            sgd_error.append(float(row[0]))
    print(sgd_error)
    train_model = train(init_flag, num_epoch, learning_rate, train_x_input, train_y_onehot, hidden_units, x_valid,
                        y_valid)
    train_model.train()
    valid_entropy_error = train_model.valid_entropy_error
    x_axis = [i for i in range(num_epoch)]
    plt.plot(x_axis, valid_entropy_error, label="Average validation cross-entropy error with adagrad")
    plt.plot(x_axis, sgd_error, label="Average validation cross-entropy error without adagrad")
    # plt.plot(x_axis, likelihood3, label="alpha = 0.1")
    plt.xlabel("epoch")  # x label
    plt.ylabel("Cross-entropy error")  # y label
    plt.legend()
    plt.show()
    '''
    '''
    #3.2
    learning_rate_list = [0.1, 0.01, 0.001]
    #for item in learning_rate_list:
    train_model = train(init_flag, num_epoch, 0.001, train_x_input, train_y_onehot, hidden_units, x_valid,
                            y_valid)
    train_model.train()
    train_entropy_error = train_model.train_entropy_error
    valid_entropy_error = train_model.valid_entropy_error
    x_axis = [i for i in range(num_epoch)]
    plt.plot(x_axis, train_entropy_error, label="Average training cross-entropy error")
    plt.plot(x_axis, valid_entropy_error, label="Average validation cross-entropy error")
    plt.xlabel("epoch")  # x label
    plt.ylabel("Cross-entropy error")  # y label
    plt.legend()
    plt.show()
    '''
    '''
    train_model = train(init_flag, num_epoch, learning_rate, train_x_input, train_y_onehot, hidden_units, x_valid, y_valid)
    train_model.train()
    #print out
    train_entropy_error = train_model.train_entropy_error
    train_label = train_model.trainout_label
    valid_entropy_error = train_model.valid_entropy_error
    valid_label = train_model.validout_label
    train_error = train_model.train_error_rate
    valid_error = train_model.valid_error_rate
    '''
    '''
    
    ##write files
    with open("%s" % (trainout), "w") as file1:
        # Writing data to a file
        for item in train_label:
            file1.write(str(item) + '\n')
    with open("%s" % (validout), "w") as file1:
        # Writing data to a file
        for item in valid_label:
            file1.write(str(item) + '\n')
    with open("%s" % (metricout), "w") as file1:
        # Writing data to a file
        for i in range(num_epoch):
            file1.write("epoch=" + str(i+1) +" crossentropy(train): " + str(train_entropy_error[i][0][0]) + '\n')
            file1.write("epoch=" + str(i+1) +" crossentropy(validation): " + str(valid_entropy_error[i][0][0]) + '\n')
        file1.write("error(train): " + str(train_error) + '\n')
        file1.write("error(validation): " + str(valid_error) + '\n')
    #print(train_model.train_entropy_error)
    #print(train_model.valid_entropy_error)
    #print(train_model.train_error_rate)
    #print(train_model.valid_error_rate)
    '''






