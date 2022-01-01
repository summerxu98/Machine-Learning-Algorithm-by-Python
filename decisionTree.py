import math
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

class TreeNode(object):
     def __init__(self, depth = set(), split = 0, data = None, left=None, right=None):
         #depth record all the split number have been used
         self.depth = depth
         self.split = split
         self.data = data
         self.left = left
         self.right = right
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


#input 1d array, output entropy
def cal_entropy(data):
    total_num = len(data)
    if(total_num == 0):return 0
    positive = 0
    negative = 0
    for item in data:
        if item == data[0]:
            positive = positive+1
        else:
            negative = negative+1
    p_posi = positive/total_num
    p_nega = negative/total_num
    if(p_posi != 0 and p_nega != 0):
        res = -p_posi * math.log(p_posi,2) - p_nega * math.log(p_nega, 2)
    elif(p_posi != 0 and p_nega == 0):
        res = -p_posi * math.log(p_posi, 2)
    elif(p_posi == 0 and p_nega != 0):
        res = - p_nega * math.log(p_nega, 2)
    else:
        res = 0
    return res

# input two 1d array
def cal_m_info(label, attri):
    #elaborate attribute
    attri1 = attri[0]
    attri2 = " "
    for item in attri:
        if(item != attri1):
            attri2 = item
    if(attri2 == " "): return 0
    #cal_possibility
    num_attri1 = 0
    num_attri2 = 0
    for item in attri:
        if(item == attri1):
            num_attri1 = num_attri1+1
        else:
            num_attri2 = num_attri2+1
    p_attri1 = num_attri1/len(attri)
    p_attri2 = num_attri2/len(attri)
    y_attri1 = label[attri == attri1]
    y_attri2 = label[attri == attri2]
    res = cal_entropy(label) - p_attri1 * cal_entropy(y_attri1) - p_attri2 * cal_entropy(y_attri2)
    return res


def choose_split(node):
    #print(node.depth)

    #print("choose split")
    max_mu_info = 0
    split = -1
    for i in range(len(node.data[0])-1):
        if(i in node.depth):
            continue
        else:
            tmp_info = cal_m_info(node.data[:,-1], node.data[:,i])
            if(tmp_info > max_mu_info):
                max_mu_info = tmp_info
                split = i
    #print(split)
    return split


#train decision stamp
##train data with row 0
def traintree(node, max_depth, traindata):
    #write label
    label_res = write_label(traindata[1:, -1])
    label_name_posi = label_res[0]
    label_name_nega = label_res[1]
    #write feature
    feature = traindata[0,:]
    ###print(node.data)
    ##empty dataset
    if (len(node.data) == 0): return
    ## come to max_depth or all the attribute have been used
    if (len(node.depth) == max_depth or len(node.depth) == len(node.data[0])-1):return
    ##pure dataset
    if(test_pure(node.data[:,-1])): return
    ##mutual info

    #print data posi and nega
    #print(write_posi_nega(node.data))
    node.split = choose_split(node)
    if(node.split == -1): return
    node.depth.add(node.split)
    #print(node.depth)
    #write how to split
    string1 = "|"
    number = len(node.depth)
    while(number!=1):
        string1 = string1 + " |"
        number = number-1
    feature_name = feature[node.split]
    res_string = string1 + " " + feature_name + " = "
    #split data

    splitdata = node.data[:,node.split]
    ##write split attribute
    attri_res = write_label(splitdata)
    attri_posi = attri_res[0]
    attri_nega = attri_res[1]
    leftdata = node.data[splitdata == attri_posi]
    rightdata = node.data[splitdata != attri_posi]
    #recurse on left and right
    ###print("train on left")
    #print(len(leftdata))
    node.left = TreeNode(depth=node.depth, split= node.split, data = leftdata)
    print(res_string + str(attri_posi) + ": " + write_posi_nega(node.left.data, label_name_posi, label_name_nega))
    traintree(node.left, max_depth, traindata)


    ###print("train on right")
    #print(len(rightdata))
    node.right = TreeNode(depth=node.depth, split= node.split, data=rightdata)
    print(res_string + str(attri_nega) + ": " + write_posi_nega(node.right.data, label_name_posi,label_name_nega))
    traintree(node.right, max_depth, traindata)
    #print(node.split)

    node.depth.remove(node.split)
    #print(node.depth)


#input 1d array
def test_pure(data):
    for item in data:
        if(item == data[0]):continue
        else: return False
    return True


#train decision tree
def train(traindata, max_depth):
    root = TreeNode(depth = set(), data = traindata[1:,:])
    traintree(root, max_depth, traindata)
    return root

#input node, predict nd array
def testnode(trainnode, predict):
    predictnode = trainnode
    #print(predictnode.left.right.data)
    while (predictnode is not None):
        # judge_split
        pre_split = predictnode.split
        # print(predictnode.left.data)
        if (predictnode.left is not None and predictnode.right is not None):
            if (predict[pre_split] == predictnode.left.data[0, pre_split]):
                predictnode = predictnode.left
            else:
                predictnode = predictnode.right
        elif (predictnode.left is None and predictnode.right is not None):
            if (predict[pre_split] == predictnode.right.data[0, pre_split]):
                predictnode = predictnode.right
            else:
                break
        elif (predictnode.right is None and predictnode.left is not None):
            if (predict[pre_split] == predictnode.left.data[0, pre_split]):
                predictnode = predictnode.right
            else:
                break
        else:
            break
    test_res_data = predictnode.data[:,-1]
    label1 = test_res_data[0]
    num1 = 0
    num2 = 0
    for item in test_res_data:
        if(item == label1):
            num1 = num1+1
        else:
            label2 = item
            num2 = num2+1
    if(num1 > num2):return label1
    elif(num1 < num2):return label2
    else:
        if(label1 < label2): return label2
        else: return label1


def test(trainnode, testarray):
    res_list = []
    for i in range(len(testarray)):
        res_list.append(testnode(trainnode, testarray[i]))
    res_list = np.array(res_list)
    return res_list

def cal_error_rate(truedata, traindata):
    n = len(truedata)
    error = 0
    for i in range(0, n):
        if(truedata[i] == traindata[i]):continue
        else: error = error+1
    return error/n


def write_label(labeldata):
    res = []
    label1 = labeldata[0]
   # label2 = " "
    res.append(label1)
    for item in labeldata:
        if (item != label1):
            label2 = item
            break
    res.append(label2)
    #print(type(res))
    return res

def cal_posi_nega(labeldata, label):
    label_num = 0
    for item in labeldata:
        if(item == label):
            label_num = label_num+1
    return label_num

def write_posi_nega(traindata, label_name_posi, label_name_nega):
    posi_num = cal_posi_nega(traindata[:, -1], label_name_posi)
    nega_num = cal_posi_nega(traindata[:, -1], label_name_nega)
    res = "[" + str(posi_num) + " " + str(label_name_posi) + "/" + str(nega_num) + " " + str(label_name_nega) + "]"
    return res

if __name__ == "__main__":

    trainfile = "politicians_train.tsv"
    testfile = "politicians_test.tsv"

    

    '''
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    maxdepth = sys.argv[3]
    maxdepth = int(maxdepth)
    trainout = sys.argv[4]
    testout = sys.argv[5]
    metricsout = sys.argv[6]
    '''

    traindata = processdata(trainfile)
    testdata = processdata(testfile)
    #maxdepth = 2

    ##create the plot
    depth = []
    train_error = []
    test_error = []
    for maxdepth in range(0, len(traindata[0])):
        depth.append(maxdepth)
        ##print tree
        res_label = write_label(traindata[1:, -1])
        print(write_posi_nega(traindata[1:, :], res_label[0], res_label[1]))
        # train
        trainnode = train(traindata, maxdepth)
        # train_data_test
        train_res_list = []
        for i in range(1, len(traindata)):
            train_res_list.append(testnode(trainnode, traindata[i]))
        train_res_list = np.array(train_res_list)
        # train_test_data
        test_res_list = []
        for i in range(1, len(testdata)):
            test_res_list.append(testnode(trainnode, testdata[i]))
        test_res_list = np.array(test_res_list)
        # cal_error_rate
        train_error_rate = cal_error_rate(traindata[1:, -1], train_res_list)
        test_error_rate = cal_error_rate(testdata[1:, -1], test_res_list)
        train_error.append(train_error_rate)
        test_error.append(test_error_rate)

    depth = np.array(depth)
    train_error = np.array(train_error)
    test_error = np.array(test_error)
    print(depth)
    print(train_error)
    print(test_error)
    plt.scatter(depth, train_error, label = 'Train Error')
    plt.scatter(depth, test_error, label = 'Test Error')
    plt.xlabel("depth")  # x label
    plt.ylabel("error rate")  # y label
    plt.legend()
    plt.show()


    '''
    ##print tree
    res_label = write_label(traindata[1:, -1])
    print(write_posi_nega(traindata[1:, :], res_label[0], res_label[1]))
    #train
    trainnode = train(traindata, maxdepth)
    #train_data_test
    train_res_list = []
    for i in range(1, len(traindata)):
        train_res_list.append(testnode(trainnode, traindata[i]))
    train_res_list = np.array(train_res_list)
    #train_test_data
    test_res_list = []
    for i in range(1, len(testdata)):
        test_res_list.append(testnode(trainnode, testdata[i]))
    test_res_list = np.array(test_res_list)
    #cal_error_rate
    train_error_rate = cal_error_rate(traindata[1:,-1], train_res_list)
    test_error_rate = cal_error_rate(testdata[1:,-1], test_res_list)

    print(train_res_list)
    print(test_res_list)
    print(train_error_rate)
    print(test_error_rate)
    '''

    '''
    #output
    with open("%s" %(trainout), "w") as file1:
        # Writing data to a file
        for item in train_res_list:
            file1.write(item + '\n')
    with open("%s" % (testout), "w") as file1:
        # Writing data to a file
        for item in test_res_list:
            file1.write(item + '\n')
    with open("%s" % (metricsout), "w") as file1:
        # Writing data to a file
        file1.write("error(train): %f" % train_error_rate + '\n')
        file1.write("error(test): %f" % test_error_rate + '\n')
    '''








