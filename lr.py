import math
import numpy as np
import csv
import math
import sys

def read_formatted_data(formatted_data):
    y_label = []
    design_X = []
    with open(formatted_data, newline='') as tsvfile:
        spamreader = csv.reader(tsvfile, delimiter="\t", quotechar='"')
        for row in spamreader:
            #print(str(row).split(",")[0])
            #print(row[0])
            y_label.append(row[0])
            #append bias term
            design_xi = np.array(row[1:])
            design_xi = np.insert(design_xi,0,1)
            design_X.append(design_xi)
    return y_label, design_X

def calculate_i_gradient(theta, designX, i, y_label, N):
    #begin math
    linear_res = designX[i].dot(theta)
    exp = math.exp(linear_res)
    gradient_constant = y_label[i] - (exp)/(1 + exp)
    return (gradient_constant/N)*designX[i]

def calculate_parameter(theta, epoch, designX, y_label, alpha, N):
    for i in range(epoch):
        #print(i)
        for j in range(N):
            gradient_update = calculate_i_gradient(theta, designX, j, y_label, N)
            #print(gradient_update)
            theta = theta + alpha * gradient_update

    #print(theta)
    return theta

def calculate_test(designX, theta, i):
    linear = designX[i].dot(theta)
    #print(linear)
    exp = math.exp(linear)
    res = exp/(1+exp)
    #print(res)
    if(res >= 0.5):
        return 1
    else:
        return 0

#return (int) result list
def calculate_test_res(designX, theta):
    res_list = []
    for i in range(len(designX)):
        res = calculate_test(designX, theta, i)
        #print(res)
        res_list.append(res)
    return res_list

def traindata(trainfile, epoch, learning_rate):
    y_label, design_X = read_formatted_data(trainfile)
    y_label = np.array(y_label, dtype=float)
    design_X = np.array(design_X, dtype=float)
    N = len(design_X)
    theta = np.zeros(len(design_X[1]))
    theta_new = calculate_parameter(theta, epoch, design_X, y_label, learning_rate, N)
    return theta_new

def testdata(testfile, theta):
    y_test_label, test_design_X = read_formatted_data(testfile)
    y_test_label = np.array(y_test_label, dtype=float)
    test_design_X = np.array(test_design_X, dtype=float)
    test_list = calculate_test_res(test_design_X, theta)
    error = 0
    for i in range(len(test_list)):
        if(test_list[i] != y_test_label[i]):
            error = error+1
    return test_list, error/len(test_list)

'''
#train_formatted_data = "./largeoutput/model2_formatted_train.tsv"
train_formatted_data = "./try_large_output/2_model_train.tsv"
#formatted_data = "./smalloutput/1_model_test.tsv"
theta_new = traindata(train_formatted_data, 500, 0.01)
test_formatted_data = "./try_large_output/2_model_test.tsv"
error, error_rate = testdata(test_formatted_data, theta_new)
print(error_rate)
python lr.py ./try_large_output/1_model_train.tsv ./try_large_output/1_model_validation.tsv ./try_large_output/1_model_test.tsv dict.txt ./try_large_output/1_train_out.labels ./try_large_output/1_test_out.labels ./try_large_output/1_metrics_out.txt 500
'''

if __name__ == "__main__":
    trainfile = sys.argv[1]
    validationfile = sys.argv[2]
    testfile = sys.argv[3]
    dict_input = sys.argv[4]
    train_out_labels = sys.argv[5]
    test_out_labels = sys.argv[6]
    matrics_out_labels = sys.argv[7]
    epoch = sys.argv[8]

    learning_rate = 0.01
    theta_new = traindata(trainfile, int(epoch), float(learning_rate))
    train_list, train_error = testdata(trainfile, theta_new)
    test_list, test_error = testdata(testfile, theta_new)
    # output
    with open("%s" % (train_out_labels), "w") as file1:
        # Writing data to a file
        for item in train_list:
            file1.write(str(item) + '\n')
    with open("%s" % (test_out_labels), "w") as file1:
        # Writing data to a file
        for item in test_list:
            file1.write(str(item) + '\n')
    with open("%s" % (matrics_out_labels), "w") as file1:
        # Writing data to a file
        file1.write("error(train): %f" % train_error + '\n')
        file1.write("error(test): %f" % test_error + '\n')

