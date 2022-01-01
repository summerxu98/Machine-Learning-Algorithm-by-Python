import sys
import numpy as np

class state:
    def __init__(self, word_index, tag_index):
        self.word = word_index
        self.hidden_state = tag_index

class input_process:
    def init_word_dict(self, index_to_word):
        self.word_dict = {}
        file = open(index_to_word, encoding='utf-8')
        i = 0
        for row in file:
            row = row.split('\n')[0]
            self.word_dict[row] = i
            i = i + 1
        self.word_length = len(self.word_dict)
    def init_tag_dict(self, index_to_tag):
        self.tag_dict = {}
        file = open(index_to_tag, encoding='utf-8')
        i = 0
        for row in file:
            row = row.split('\n')[0]
            self.tag_dict[row] = i
            i = i + 1
        self.tag_length = len(self.tag_dict)
    def process_word(self, wordfile):
        self.data = []
        file = open(wordfile)
        sequence = []
        for row in file:
            row = row.split('\n')[0]
            if (len(row) != 0):
                row = row.split('\t')
                word_index = self.word_dict[row[0]]
                tag_index = self.tag_dict[row[1]]
                sequence.append(state(word_index, tag_index))
            else:
                self.data.append(sequence)
                sequence = []
        self.data.append(sequence)
        self.data = np.array(self.data, dtype=object)

class calculate_prob:
    def __init__(self, data, tag_length, word_length):
        self.data = data
        self.init_prob = np.zeros((tag_length,1))
        self.emis_prob = np.zeros((tag_length, word_length))
        self.trans_prob = np.zeros((tag_length, tag_length))
    def cal_init(self):
        for i in range(len(self.data)):
            tag_tmp =  self.data[i][0].hidden_state
            self.init_prob[tag_tmp][0] +=  1
        self.init_prob = np.add(self.init_prob, np.ones((len(self.init_prob), 1)))
        init_sum = np.sum(self.init_prob)
        self.init_prob = self.init_prob/init_sum
    def cal_emis(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                tag_tmp = self.data[i][j].hidden_state
                word_tmp = self.data[i][j].word
                self.emis_prob[tag_tmp][word_tmp] += 1
        self.emis_prob = np.add(self.emis_prob, np.ones((len(self.emis_prob), len(self.emis_prob[0]))))
        emis_sum = np.sum(self.emis_prob, axis = 1, keepdims = True)
        self.emis_prob = np.divide(self.emis_prob, emis_sum)
    def cal_trans(self):
        for i in range(len(self.data)):
            for j in range(1, len(self.data[i])):
                tag_last = self.data[i][j-1].hidden_state
                tag_cur = self.data[i][j].hidden_state
                self.trans_prob[tag_last][tag_cur] += 1
        self.trans_prob = np.add(self.trans_prob, np.ones((len(self.trans_prob), len(self.trans_prob[0]))))
        print(self.trans_prob)
        trans_sum = np.sum(self.trans_prob, axis = 1, keepdims = True)
        self.trans_prob = np.divide(self.trans_prob, trans_sum)


if __name__ == '__main__':


    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    input_process = input_process()
    input_process.init_word_dict(index_to_word)
    input_process.init_tag_dict(index_to_tag)
    input_process.process_word(train_input)
    prob = calculate_prob(input_process.data, input_process.tag_length, input_process.word_length)
    prob.cal_init()
    prob.cal_emis()
    prob.cal_trans()
    np.savetxt(hmminit, prob.init_prob, fmt='%.18e', delimiter=' ', newline='\n')
    np.savetxt(hmmemit, prob.emis_prob, fmt='%.18e', delimiter=' ', newline='\n')
    np.savetxt(hmmtrans, prob.trans_prob, fmt='%.18e', delimiter=' ', newline='\n')


'''
    input_process = input_process()
    input_process.init_word_dict('toy_data/index_to_word.txt')
    input_process.init_tag_dict('toy_data/index_to_tag.txt')
    input_process.process_word('toy_data/train.txt')
    print(input_process.tag_length)
    prob = calculate_prob(input_process.data, input_process.tag_length, input_process.word_length)
    prob.cal_init()
    print(prob.init_prob)
    #np.savetxt('toy_data/init_out.txt', prob.init_prob, fmt='%.18e', delimiter=' ', newline='\n')
    prob.cal_emis()
    print(prob.emis_prob)
    prob.cal_trans()
    print(prob.trans_prob)
    '''




#python3 learnhmm.py en_data/train.txt en_data/index_to_word.txt en_data/index_to_tag.txt en_data/init_out.txt en_data/emit_out.txt en_data/trans_out.txt
#python3 learnhmm.py toy_data/train.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt toy_data/init_out.txt toy_data/emit_out.txt toy_data/trans_out.txt