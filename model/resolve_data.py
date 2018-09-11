#-*-coding:utf-8-*-
import sys
import os
import numpy as np
from setting import *
label = []
tree = []
nl = []
node = []
parent_node = []
parent_list = []
rule_list = []
func_list = []

t_label = []
t_tree = []
t_nl = []
t_func_list = []
t_node = []
t_parent_node = []
t_parent_list = []
t_rule_list = []


outputtreevoc = []
outputvoc = []

train_nl = np.zeros([10, 10])
train_tree = np.zeros([10, 10])
#train_deep = np.zeros([10,10])
#train_func = np.zeros([10,10])
train_label = np.zeros([10, 10])
train_node = np.zeros([10, 10])
train_parent = np.zeros([10, 10])
train_parentlist = np.zeros([10,10])
train_rulelist = np.zeros([10,10])
train_funclist = np.zeros([10,10])
train_copy = np.zeros([10,10])

test_nl = np.zeros([10, 10])
test_tree = np.zeros([10, 10])
#test_deep = np.zeros([10,10])
#test_func = np.zeros([10,10])
test_label = np.zeros([10, 10])
test_node = np.zeros([10, 10])
test_parent = np.zeros([10, 10])
test_parentlist = np.zeros([10,10])
test_rulelist = np.zeros([10,10])
test_funclist = np.zeros([10,10])
test_copy = np.zeros([10,10])

nl_train = []
nl_test = []

#classnum = readrule()
vocabulary = {}
tree_vocabulary = {}
Rule = []
#def readrule_data:
#f = open("Rule.txt", "r")
#lines = f.readlines()
#f.close()
#for line in lines:
#    if len(line.split()) <= 2:
#        Rule.append(line.split()[0])
#    else:
#        Rule.append(line.split()[2])


def load_vocabulary():
    global vocabulary
    global tree_vocabulary
    if os.path.exists("vocabulary.txt"):
        f = open("vocabulary.txt", "r")
        lines = f.readline()
        vocabulary = eval(lines)
        print(vocabulary)
    if os.path.exists("tree_vocabulary.txt"):
        f = open("tree_vocabulary.txt", "r")
        lines = f.readline()
        tree_vocabulary = eval(lines)
def sentence2vec(sentence):
    words = sentence.split()
    res = []
    for x in words:
        if x not in vocabulary:
            vocabulary[x] = len(vocabulary)
            outputvoc.append(x)
        #    print (x)
            res.append(vocabulary[x])             
        else:
            res.append(vocabulary[x])
    return res

def tree2vec(sentence):
    words = sentence.split()
    res = []
    #if len(sentence) < 3:
        #print(sentence + "1")
        #sys.exit(0)
    if sentence == "" or sentence == " ":
#        print("empty_line")
        return res
    for x in words:
        if x not in tree_vocabulary:
            tree_vocabulary[x] = len(tree_vocabulary)
            res.append(tree_vocabulary[x])
            outputtreevoc.append(x)
        else:
            res.append(tree_vocabulary[x])
    return res


def load_data():
    global vocabulary
    global tree_vocabulary
    f = open("train.txt", "r")
    lines = f.readlines()
    for i in range(int(len(lines) / 8)):
        nl_line = lines[8 * i][:-1]
        if len(nl_line.split())>=150:
            print(nl_line)
            continue 
        nl_train.append(nl_line.split())
        tree_line = lines[8 * i + 1][:-1]
        label_line = lines[8 * i + 6][:-1]
        rule_list_line = lines[8 * i + 5][:-1]
        node_line = lines[8 * i + 2][:-1]
        parent_node_line = lines[8 * i + 3][:-1]
        #deep_line = lines[8 * i + 4][:-1]
        func_line = lines[8 * i + 7][:-1]
        parent_list_line = lines[8 * i + 4][:-1]
        #print(nl_line, tree_line, label_line)
        nl.append(sentence2vec(nl_line))
        tree.append(tree2vec(tree_line))
        node.append(tree2vec(node_line))
        parent_node.append(tree2vec(parent_node_line))
        try:
            label.append(int(label_line))
        except:
            print("error", nl_line, tree_line, label_line)
        rl = str(rule_list_line).split()
        ress = []
        for r in rl:
            ress.append(int(r))
        rule_list.append(ress)
        func_list.append(tree2vec(func_line))
        #deep.append(int(deep_line))
        parent_list.append(tree2vec(parent_list_line))
        #if func_line in tree_vocabulary:
        #    func.append(tree_vocabulary[func_line])
        #else:
        #    func.append(0)
        # if node_line in tree_vocabulary:
        #     node.append(tree_vocabulary[node_line])
        # else:
        #     node.append(0)
        # if parent_node_line in tree_vocabulary:
        #     parent_node.append(tree_vocabulary[parent_node_line])
        # else:
        #     parent_node.append(0)

    f = open("dev.txt", "r")
    lines = f.readlines()
    for i in range(int(len(lines) / 8)):
        nl_line = lines[8 * i][:-1]
        if len(nl_line.split())>=150:
            print(nl_line)
            return
        nl_test.append(nl_line.split())
        tree_line = lines[8 * i + 1][:-1]
        label_line = lines[8 * i + 6][:-1]
        rule_list_line = lines[8 * i + 5][:-1]
        node_line = lines[8 * i + 2][:-1]
        parent_node_line = lines[8 * i + 3][:-1]
        # deep_line = lines[8 * i + 4][:-1]
        func_line = lines[8 * i + 7][:-1]
        parent_list_line = lines[8 * i + 4][:-1]
        #print(nl_line, tree_line, label_line)
        sentence2vec(nl_line)
        tree2vec(tree_line)
        tree2vec(node_line)
        tree2vec(parent_node_line)
        tree2vec(func_line)
        tree2vec(parent_list_line)


    f = open("test.txt", "r")
    lines = f.readlines()
    for i in range(int(len(lines) / 8)):
        nl_line = lines[8 * i][:-1]
        if len(nl_line.split())>=150:
            print(nl_line)
            return
        nl_test.append(nl_line.split())
        tree_line = lines[8 * i + 1][:-1]
        label_line = lines[8 * i + 6][:-1]
        rule_list_line = lines[8 * i + 5][:-1]
        node_line = lines[8 * i + 2][:-1]
        parent_node_line = lines[8 * i + 3][:-1]
        # deep_line = lines[8 * i + 4][:-1]
        func_line = lines[8 * i + 7][:-1]
        parent_list_line = lines[8 * i + 4][:-1]
        #print(nl_line, tree_line, label_line)
        sentence2vec(nl_line)
        tree2vec(tree_line)
        tree2vec(node_line)
        tree2vec(parent_node_line)
        tree2vec(func_line)
        tree2vec(parent_list_line)
        
        
    f = open(sys.argv[2] + ".txt", "r")
    lines = f.readlines()
    for i in range(int(len(lines) / 8)):
        nl_line = lines[8 * i][:-1]
        if len(nl_line.split())>=150:
            print(nl_line)
            return
        nl_test.append(nl_line.split())
        tree_line = lines[8 * i + 1][:-1]
        label_line = lines[8 * i + 6][:-1]
        rule_list_line = lines[8 * i + 5][:-1]
        node_line = lines[8 * i + 2][:-1]
        parent_node_line = lines[8 * i + 3][:-1]
        # deep_line = lines[8 * i + 4][:-1]
        func_line = lines[8 * i + 7][:-1]
        parent_list_line = lines[8 * i + 4][:-1]
        #print(nl_line, tree_line, label_line)
        t_nl.append(sentence2vec(nl_line))
        t_tree.append(tree2vec(tree_line))
        try:
            t_label.append(int(label_line))
        except:
            print(nl_line, tree_line, label_line)
        t_node.append(tree2vec(node_line))
        t_parent_node.append(tree2vec(parent_node_line))
        rl = str(rule_list_line).split()
        ress = []
        for r in rl:
            try:
                ress.append(int(r))
            except:
       #         print(nl_line, tree_line, label_line)
                break
        t_rule_list.append(ress)
        t_func_list.append(tree2vec(func_line))
        #t_deep.append(int(deep_line))
        t_parent_list.append(tree2vec(parent_list_line))
        #if func_line in tree_vocabulary:
        #    t_func.append(tree_vocabulary[func_line])
        #else:
        #    t_func.append(0)
        # if node_line in tree_vocabulary:
        #     t_node.append(tree_vocabulary[node_line])
        # else:
        #     t_node.append(0)
        # if parent_node_line in tree_vocabulary:
        #     t_parent_node.append(tree_vocabulary[parent_node_line])
        # else:
        #     t_parent_node.append(0)

def data2numpy():
    global train_nl
    global train_tree
    global train_label
    global train_node
    global train_parent
    global train_parentlist
    global train_rulelist
    global train_funclist
    global train_copy
    global test_nl
    global test_tree
    global test_label
    global test_node
    global test_parent
    global test_parentlist
    global test_rulelist
    global test_funclist
    global test_copy
    nl_set = np.zeros([len(nl), nl_len])
    tree_set = np.zeros([len(nl), tree_len])
    label_set = np.zeros([len(nl), classnum])
    node_set = np.zeros([len(nl), tree_len])
    parent_node_set = np.zeros([len(nl), tree_len])
    parent_list_set = np.zeros([len(nl), parent_len])
    rule_list_set = np.zeros([len(nl), rulelist_len])
    func_list_set = np.zeros([len(nl), 10])
    #copy_set = np.zeros([len(nl), len(Rule), nl_len])

    for i in range(len(nl)):
        for j in range(len(nl[i])):
            #print(nl[i])
            nl_set[i, j] = nl[i][j]
    for i in range(len(tree)):
        for j in range(len(tree[i])):
            if j == tree_len:
                print(i)
            tree_set[i, j] = tree[i][j]
            node_set[i, j] = node[i][j]
            parent_node_set[i, j] = parent_node[i][j]
    for i in range(len(parent_list)):
        for j in range(len(parent_list[i])):
            parent_list_set[i, j] = parent_list[i][j]
    for i in range(len(func_list)):
        for j in range(len(func_list[i])):
            func_list_set[i, j] = func_list[i][j]
    for i in range(len(label)):
        label_set[i, label[i]] = 1
    for i in range(len(rule_list)):
        #print (rule_list[i])
        for j in range(len(rule_list[i])):
            rule_list_set[i, j] = rule_list[i][j]
    #for i in range(len(nl)):
        #print (rule_list[i])
    #    for j in range(len(Rule)):
    #        for k in range(nl_len):
    #            if k >= len(nl_train[i]):
    #                break
    #            if Rule[j] == nl_train[i][k]:
    #                copy_set[i, j, k] = 1

    train_nl = nl_set
    train_tree = tree_set
    train_label = label_set
    train_node = node_set
    train_parent = parent_node_set
    train_parentlist = parent_list_set
    train_rulelist = rule_list_set
    train_funclist = func_list_set
    #train_copy = copy_set
    
    
    test_nl_set = np.zeros([len(t_nl), nl_len])
    test_tree_set = np.zeros([len(t_nl), tree_len])
    test_label_set = np.zeros([len(t_nl), classnum])
    test_node_set = np.zeros([len(t_nl), tree_len])
    test_parent_node_set = np.zeros([len(t_nl), tree_len])
    test_parent_list_set = np.zeros([len(t_nl), parent_len])
    test_rule_list_set = np.zeros([len(t_nl), rulelist_len]) #62758622
    test_func_list_set = np.zeros([len(t_nl), 10])
    #test_copy_set = np.zeros([len(t_nl), len(Rule), nl_len])
    
    for i in range(len(t_nl)):
        for j in range(len(t_nl[i])):
            test_nl_set[i, j] = t_nl[i][j]
    for i in range(len(t_tree)):
        for j in range(len(t_tree[i])):
            test_tree_set[i, j] = t_tree[i][j]
            test_node_set[i, j] = t_node[i][j]
            test_parent_node_set[i, j] = t_parent_node[i][j]
    for i in range(len(t_parent_list)):
        for j in range(len(t_parent_list[i])):
            test_parent_list_set[i, j] = t_parent_list[i][j]
    for i in range(len(t_func_list)):
        for j in range(len(t_func_list[i])):
            test_func_list_set[i, j] = t_func_list[i][j]
    for i in range(len(t_label)):
        test_label_set[i, t_label[i]] = 1
    for i in range(len(t_rule_list)):
        for j in range(len(t_rule_list[i])):
            test_rule_list_set[i, j] = t_rule_list[i][j]
    #for i in range(len(t_nl)):
        #print (rule_list[i])
    #    for j in range(len(Rule)):
    #        for k in range(nl_len):
    #            if k >= len(nl_test[i]):
    #                break
    #            if Rule[j] == nl_test[i][k]:
    #                test_copy_set[i, j, k] = 1

    test_nl = test_nl_set
    test_tree = test_tree_set
    test_label = test_label_set
    test_node = test_node_set
    test_parent = test_parent_node_set
    test_parentlist = test_parent_list_set
    test_rulelist = test_rule_list_set
    test_funclist = test_func_list_set
    #test_copy = test_copy_set

def get_train_batch(batch_size):
    global train_nl
    global train_tree
    global train_label
    global train_node
    global train_parent
    global train_parentlist
    global train_rulelist
    global train_funclist
    global train_copy
    batch_num = int(int(train_nl.shape[0]) / batch_size)
    num_lst = np.random.permutation(range(int(train_nl.shape[0])))
    batch0 = []
    batch1 = []
    batch2 = []
    batch3 = []
    batch4 = []
    batch5 = []
    batch6 = []
    batch7 = []
    batch8 = []
    batch_0 = train_nl[num_lst]
    batch_1 = train_tree[num_lst]
    batch_2 = train_node[num_lst]
    batch_3 = train_parent[num_lst]
    batch_4 = train_label[num_lst]
    batch_5 = train_parentlist[num_lst]
    batch_6 = train_rulelist[num_lst]
    batch_7 = train_funclist[num_lst]
    #batch_8 = train_copy[num_lst]
    for i in range(batch_num):
        batch0.append(batch_0[batch_size * i:batch_size * i + batch_size, :])
        batch1.append(batch_1[batch_size * i:batch_size * i + batch_size, :])
        batch2.append(batch_2[batch_size * i:batch_size * i + batch_size, :])
        batch3.append(batch_3[batch_size * i:batch_size * i + batch_size, :])
        batch4.append(batch_4[batch_size * i:batch_size * i + batch_size, :])
        batch5.append(batch_5[batch_size * i:batch_size * i + batch_size, :])
        batch6.append(batch_6[batch_size * i:batch_size * i + batch_size, :])
        batch7.append(batch_7[batch_size * i:batch_size * i + batch_size, :])
     #   batch8.append(batch_8[batch_size * i:batch_size * i + batch_size, :])
        
    '''batch0.append(batch_0[batch_size * batch_num:, :])
    batch1.append(batch_1[batch_size * batch_num:, :])
    batch2.append(batch_2[batch_size * batch_num:, :])'''
    return batch0, batch1, batch2, batch3, batch4 , batch5, batch6, batch7#, batch8

def get_test_batch():
    return test_nl, test_tree, test_node, test_parent, test_label ,test_parentlist, test_rulelist, test_funclist#, test_copy

def get_valid_batch():
    batch0 = []
    batch1 = []
    batch2 = []
    batch3 = []
    batch4 = []
    batch5 = []
    batch6 = []
    batch7 = []
    batch8 = []
    batch_size = 500
    bound = int(test_nl.shape[0]/batch_size)
    for i in range(bound):
        batch0.append(test_nl[batch_size * i:batch_size * i + batch_size, :])
        batch1.append(test_tree[batch_size * i:batch_size * i + batch_size, :])
        batch2.append(test_node[batch_size * i:batch_size * i + batch_size, :])
        batch3.append(test_parent[batch_size * i:batch_size * i + batch_size, :])
        batch4.append(test_label[batch_size * i:batch_size * i + batch_size, :])
        batch5.append(test_parentlist[batch_size * i:batch_size * i + batch_size, :])
        batch6.append(test_rulelist[batch_size * i:batch_size * i + batch_size, :])
        batch7.append(test_funclist[batch_size * i:batch_size * i + batch_size, :])
    #    batch8.append(test_copy[batch_size * i:batch_size * i + batch_size, :])
    batch0.append(test_nl[batch_size * bound:, :])
    batch1.append(test_tree[batch_size * bound:, :])
    batch2.append(test_node[batch_size * bound:, :])
    batch3.append(test_parent[batch_size * bound:, :])
    batch4.append(test_label[batch_size * bound:, :])
    batch5.append(test_parentlist[batch_size * bound:, :])
    batch6.append(test_rulelist[batch_size * bound:, :])
    batch7.append(test_funclist[batch_size * bound:, :])
    #batch8.append(test_copy[batch_size * bound:, :])
    return batch0, batch1, batch2, batch3, batch4 , batch5, batch6, batch7#, batch8

def resolve_data():
    global classnum
    classnum = readrule()
    global vocabulary
    global tree_vocabulary
    vocabulary["Unknown"] = 0
    outputvoc.append("Unknown")
    tree_vocabulary["Unknown"] = 0
    outputtreevoc.append("Unknown")
    #load_vocabulary()
    load_data()
    if not os.path.exists(sys.argv[3] + "/vocabulary.txt"):
        f = open(sys.argv[3] + "/vocabulary.txt", "w")
        #vocabulary_s = sorted(vocabulary.items(), key = lambda v:v[1])
        #print (len(vocabulary))
        #print (len(outputvoc))
        #print (vocabulary)
        #for x in outputvoc:
            #f.write(x)
            #f.write("\n")
        f.write(str(vocabulary))
        f.close()
    if not os.path.exists(sys.argv[3] + "/tree_vocabulary.txt"):
        f = open(sys.argv[3] + "/tree_vocabulary.txt", "w")
        #tree_vocabulary_s = sorted(tree_vocabulary.items(), key = lambda v:v[1])
        #for x in outputtreevoc:
            #f.write(x)
            #f.write("\n")
        f.write(str(tree_vocabulary))
        f.close()
    #print(vocabulary)
    #print(tree_vocabulary)
    data2numpy()
    #get_train_batch(50)
#    print(len(test_nl))
#resolve_data()
