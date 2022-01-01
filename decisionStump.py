import csv
import numpy as np
import scipy
import sys



#datafile is tsv, convert tsv to 2d array
def processdata(datafile):
    #data list --> np.metrix
    data_list = [];
    with open(datafile, newline='') as tsvfile:
        spamreader = csv.reader(tsvfile, delimiter="\t", quotechar='"')
        for row in spamreader:
            # print(row)
            data_list.append(row)
    #return metrix
    data = np.array(data_list)
    return data


def train(traindata, splitindex):
    feature_col = traindata[1:, splitindex]
    label_col = traindata[1:, -1]
    train_res = {}
    # binary attribute value
    attr_1 = feature_col[0]
    #identify binary attribute
    for item in feature_col:
        if(item == attr_1): continue
        else:
            attr_2 = item
            break;
    # if feature = y, count
    label_y = label_col[feature_col == attr_1]
    label_1 = 0
    label_2 = 0
    y_label = label_y[0]
    #identify binary label
    for item in label_y:
        if(item == y_label): continue
        else:
            n_label = item
            break;
    #majority vote
    for item in label_y:
        if (item == y_label):
            label_1 = label_1 + 1
        else:
            label_2 = label_2 + 1
    if (label_1 > label_2):
        train_res[attr_1] = y_label
    elif(label_1 < label_2):
        train_res[attr_1] = n_label
    else:
        rand = np.random.rand()
        if(rand <= 0.5):
            train_res[attr_1] = y_label
        else:
            train_res[attr_1] = n_label
    # if feature = n, count
    label_y = label_col[feature_col == attr_2]
    label_1 = 0
    label_2 = 0
    y_label = label_y[0]
    # identify binary label
    for item in label_y:
        if (item == y_label):
            continue
        else:
            n_label = item
            break;
    # majority vote
    for item in label_y:
        if (item == y_label):
            label_1 = label_1 + 1
        else:
            label_2 = label_2 + 1
    if (label_1 > label_2):
        train_res[attr_2] = y_label
    elif (label_1 < label_2):
        train_res[attr_2] = n_label
    else:
        rand = np.random.rand()
        if (rand <= 0.5):
            train_res[attr_2] = y_label
        else:
            train_res[attr_2] = n_label
    return train_res


def test(testdata, splitindex, train_dic):
    feature_test = testdata[1:, splitindex]
    label_res = []
    for item in feature_test:
        label_res.append(train_dic[item])
    label_res = np.array(label_res)
    return label_res
#testdata = processdata("education_test.tsv")
#print(test(testdata, 0, train(traindata, 0)))

#input two arrays
def cal_error_rate(truedata, traindata):
    n = len(truedata)
    error = 0
    for i in range(0, n):
        if(truedata[i] == traindata[i]):continue
        else: error = error+1
    return error/n


if __name__ == "__main__":
    #argument
    '''
    trainfile = "education_train.tsv"
    testfile = "education_test.tsv"
    splitindex = 5
    trainout = "education_5_train.labels"
    testout = "education_5_test.labels"
    metricsout = "education_5_metrics.txt"
    '''
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    splitindex = sys.argv[3]
    splitindex = int(splitindex)
    trainout = sys.argv[4]
    testout = sys.argv[5]
    metricsout = sys.argv[6]
    #process and train, output dictinary trainres
    traindata = processdata(trainfile)
    testdata = processdata(testfile)
    trainres = train(traindata, splitindex)
    #train output
    ##train_res and test_res are numpy array
    train_res = test(traindata, splitindex, trainres)
    ##write train res
    with open("%s" %(trainout), "w") as file1:
        # Writing data to a file
        for item in train_res:
            file1.write(item + '\n')
    #test output
    test_res = test(testdata, splitindex, trainres)
    ##write test res
    with open("%s" % (testout), "w") as file1:
        # Writing data to a file
        for item in test_res:
            file1.write(item + '\n')
    #calculate error
    train_error = cal_error_rate(traindata[1:, -1], train_res)
    test_error = cal_error_rate(testdata[1:, -1], test_res)
    #write error
    with open("%s" % (metricsout), "w") as file1:
        # Writing data to a file
        file1.write("error(train): %f" % train_error + '\n')
        file1.write("error(test): %f" % test_error + '\n')
