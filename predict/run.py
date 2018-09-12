#-*-coding:utf-8-*-
import sys
# sys.path.append('..')
from code_generate_model import *
#from resolve_data import *
from code_generate_model import code_gen_model as gen_func
from code_generate_model import code_gen_model as gen_var
import os
import tensorflow as tf
import numpy as np
#import os
import queue as Q
from setting import *
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"]="0"

Rule = []

vocabu = {}
tree_vocabu = {}
vocabu_func = {}
tree_vocabu_func = {}
vocabu_var = {}
tree_vocabu_var = {}


Ori_Nl = ""

classnum = 300
embedding_size = 128
conv_layernum = 128
conv_layersize = 2
rnn_layernum = 50
batch_size = 60
NL_vocabu_size = 1
Tree_vocabu_size = 1
NL_len = nl_len
Tree_len = tree_len
learning_rate = 1e-5
keep_prob = 0.5
train_times = 1000
parent_len = 100

classnum_func = 337
embedding_size_func = 128
conv_layernum_func = 128
conv_layersize_func = 2
rnn_layernum_func = 50

#classnum_func = 330
#embedding_size_func = 256
#conv_layernum_func = 256
#conv_layersize_func = 3
#rnn_layernum_func = 50
batch_size_func = 100
NL_vocabu_size_func = 1
Tree_vocabu_size_func = 1
NL_len_func = nl_len
Tree_len_func = tree_len
learning_rate_func = 1e-4
keep_prob_func = 0.5
train_times_func = 10000

classnum_gen = 254
embedding_size_gen = 128
conv_layernum_gen = 128
conv_layersize_gen = 2
rnn_layernum_gen = 50
batch_size_gen = 100
NL_vocabu_size_gen = 1
Tree_vocabu_size_gen = 11
NL_len_gen = nl_len
Tree_len_gen = tree_len
learning_rate_gen = 1e-4
keep_prob_gen = 0.5
train_times_gen = 10000


numberstack = []
list2wordlist = []

class line2word :
    def __init__(self, f, gf, ggf, nl , ori):
        self.f = f
        self.gf = gf
        self.ggf = ggf
        self.nl = nl
        self.ori = ori
        self.used = False

def readbacklist (Nl_num):
    global numberstack
    global list2wordlist
    numberstack = []
    f = open("test/test_output_our_ast_back/"+str(Nl_num)+".txt","r")
    lines = f.readlines()
    f.close()
    list2wordlist = []
    for line in lines:
        st = str(line)[:-1].replace("_fu_nc_na_me","").split(" ")
        if len(st)==1:
            numberstack.append("9")
        else:
            word = line2word(st[0],st[1],st[2],st[3],st[4])
            list2wordlist.append(word)

def line2word_var (line,father,JavaOut):
    for l in JavaOut.list2wordlistjava:
        if l.nl == line:
            if l.f == father and l.used == False:
                l.used = True
                return l.ori

    for l in JavaOut.list2wordlistjava:
        if l.nl == line:
            if l.used == False:
                l.used = True
                return l.ori

    for l in JavaOut.list2wordlistjava:
        if l.nl == line:
            l.used = False

    for l in JavaOut.list2wordlistjava:
        if l.nl == line:
            if l.f == father and l.used == False:
                l.used = True
                return l.ori

    for l in JavaOut.list2wordlistjava:
        if l.nl == line:
            if l.used == False:
                l.used = True
                return l.ori

    return "-12345"

def getnumberstack (JavaOut):
    if len(JavaOut.numberstack) == 0 :
        return "-12345"
    now = JavaOut.numberstack.pop(0)
    JavaOut.numberstack.append(now)
    return now


def create_model_func(session):
    if(os.path.exists("../model/func")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint("../model/func"))
        print("load the model")

def create_model_var(session):
    if(os.path.exists("../model/var")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint("../model/var"))
        print("load the model")
    else:
        session.run(tf.global_variables_initializer())
        print("create a new model")

def create_model(session):
    if(os.path.exists("../model/tree")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint("../model/tree"))
        print("load the model")
    else:
        session.run(tf.global_variables_initializer())
        print("create a new model")


def save_model(session, number):
    saver = tf.train.Saver()
    saver.save(session, "save" + str(number) + "/model.cpkt")


def run():
    Code_gen_model = code_gen_model(classnum, embedding_size, conv_layernum, conv_layersize, rnn_layernum,
                                    batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, parent_len, learning_rate, keep_prob)
    valid_batch = get_valid_batch()
    best_accuracy = 0.13
   # config = tf.ConfigProto(device_count={"GPU": 0})
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    f = open("out.txt", "w")
    with tf.Session(config=config) as sess:
        create_model(sess)
        for i in range(train_times):
            batch = get_train_batch(batch_size)
            for j in range(len(batch[0]) - 1):
                if j % 400 == 0:
                    print("train " + str(i) + " echos " + str(j) + " times")
                #Code_gen_model.optim.run(session=sess, feed_dict={Code_gen_model.input_NL: batch[0][j],
                                                                  # Code_gen_model.input_Tree: batch[1][j],
                                                                  # Code_gen_model.inputY: batch[4][j],
                                                                  # Code_gen_model.inputN: batch[2][j],
                                                                  # Code_gen_model.inputP: batch[3][j],
                                                                  # Code_gen_model.inputparentlist: batch[5][j]
                                                                  # })
                if j % 500 == 0:
                    ac = Code_gen_model.accuracy.eval(session=sess,
                                                      feed_dict={Code_gen_model.input_NL: valid_batch[0],
                                                                 Code_gen_model.input_Tree: valid_batch[1],
                                                                 Code_gen_model.inputY: valid_batch[4],
                                                                 Code_gen_model.inputN: valid_batch[2],
                                                                 Code_gen_model.inputP: valid_batch[3],
                                                                 Code_gen_model.inputparentlist: valid_batch[5],
                                                                 Code_gen_model.inputrulelist:valid_batch[6],
                                                                 Code_gen_model.keep_prob: 1.0,
                                                                 Code_gen_model.is_train: False
                                                                 })
                    loss = Code_gen_model.loss.eval(session=sess,
                                                    feed_dict={Code_gen_model.input_NL: valid_batch[0],
                                                               Code_gen_model.input_Tree: valid_batch[1],
                                                               Code_gen_model.inputY: valid_batch[4],
                                                               Code_gen_model.inputN: valid_batch[2],
                                                               Code_gen_model.inputP: valid_batch[3],
                                                               Code_gen_model.inputparentlist: valid_batch[5],
                                                               Code_gen_model.inputrulelist:valid_batch[6],
                                                               Code_gen_model.keep_prob: 1.0,
                                                               Code_gen_model.is_train: False
                                                               })
                    #loss = 0.0
                    strs = str(ac) + " " + str(loss) + "\n"
                    f.write(strs)
                    print("current accuracy " +
                          str(ac) + "loss is " + str(loss))
                    if loss <= best_accuracy:
                        best_accuracy = loss
                        save_model(sess, 1)
                        print("find the better accuracy " +
                              str(best_accuracy) + "in echos " + str(i))
                 
                Code_gen_model.optim.run(session=sess, feed_dict={Code_gen_model.input_NL: batch[0][j],
                                                                  Code_gen_model.input_Tree: batch[1][j],
                                                                  Code_gen_model.inputY: batch[4][j],
                                                                  Code_gen_model.inputN: batch[2][j],
                                                                  Code_gen_model.inputP: batch[3][j],
                                                                  Code_gen_model.inputparentlist: batch[5][j],
                                                                  Code_gen_model.inputrulelist:batch[6][j],
                                                                  Code_gen_model.keep_prob: 0.5,
                                                                  Code_gen_model.is_train: True
                                                                  })
        # print(eval(tmpy, "data/q_a_test"))
    f.close()
    print("training finish")
    return



def loadvoc():
    global vocabu
    global tree_vocabu
    #f = open("vocabulary.txt", "r")
#    for x in f:
#        x = x[:-1]
#        vocabu[x] = len(vocabu)
    f = open("../model/tree/vocabulary.txt", "r")
    line = f.readline()
    vocabu = eval(line)
    f.close()
    f = open("../model/tree/tree_vocabulary.txt", "r")
#    for x in f:
#        x = x[:-1]
#        tree_vocabu[x] = len(tree_vocabu)
#    print ("tree")
    line = f.readline()
    tree_vocabu = eval (line)


def loadvoc_func():
    global vocabu_func
    global tree_vocabu_func
    f = open("../model/func/vocabulary.txt", "r")
    line = f.readline()
    vocabu_func = eval(line)
    #for x in f:
    #    x = x[:-1]
    #    vocabu_func[x] = len(vocabu_func)
    f = open("../model/func/tree_vocabulary.txt", "r")
    line = f.readline()
    tree_vocabu_func = eval (line)
    #for x in f:
    #    x = x[:-1]
    #    tree_vocabu_func[x] = len(tree_vocabu_func)

def loadvoc_var():
    global tree_vocabu_var
    global vocabu_var
    f = open("../model/var/vocabulary.txt", "r")
    line = f.readline()
    vocabu_var = eval(line)
#    for x in f:
#        x = x[:-1]
#        vocabu_var[x] = len(vocabu_var)
    f = open("../model/var/tree_vocabulary.txt", "r")
    line = f.readline()
    tree_vocabu_var = eval(line)
#    for x in f:
#        x = x[:-1]
#        tree_vocabu_var[x] = len(tree_vocabu_var)

def NL2matrix_func(NL):
    res = np.zeros([1, NL_len])
    words = NL.split()
    for i in range(len(words)):
        if words[i] in vocabu_func:
            res[0, i] = vocabu_func[words[i]]
        else:
            res[0, i] = 0
    return res

def NL2matrix_var(NL):
    res = np.zeros([1, NL_len])
    words = NL.split()
    for i in range(len(words)):
        if words[i] in vocabu_var:
            res[0, i] = vocabu_var[words[i]]
        else:
            res[0, i] = 0
    return res

def NL2matrix(NL):
    res = np.zeros([1, NL_len])
    words = NL.split()
    for i in range(len(words)):
        if words[i] in vocabu:
            res[0, i] = vocabu[words[i]]
        else:
            res[0, i] = 0
    return res

def Tree2matrix_func(NL):
    res = np.zeros([1, Tree_len])
    words = NL.split(" ")
    for i in range(len(words)):
        if words[i] in tree_vocabu_func:
            res[0, i] = tree_vocabu_func[words[i]]
    return res

def Tree2matrix_var(NL):
    res = np.zeros([1, Tree_len])
    words = NL.split(" ")
    for i in range(len(words)):
        if words[i] in tree_vocabu_var:
            res[0, i] = tree_vocabu_var[words[i]]
    return res

def Tree2matrix_func2(NL,le):
    res = np.zeros([1, le])
    words = NL
#    if len(words) == 0 :
#        res[0,0] = 336
    for i in range(min(le,len(words))):
        if words[i] in tree_vocabu_func:
            res[0, i] = tree_vocabu_func[words[i]]
    return res

def Tree2matrix_var2(NL,le):
    res = np.zeros([1, le])
    words = NL
    for i in range(len(words)):
        if i >= le:
            break
        if words[i] in tree_vocabu_var:
            res[0, i] = tree_vocabu_var[words[i]]
    return res

def Tree2matrix(NL):
    res = np.zeros([1, Tree_len])
    words = NL.split()
    for i in range(len(words)):
        if words[i] in tree_vocabu:
            res[0, i] = tree_vocabu[words[i]]
    return res

def Tree2matrix2(NL, le):
    res = np.zeros([1, le])
    words = NL
    for i in range(len(words)):
        if i >= le :
            break
        if words[i] in tree_vocabu:
            res[0, i] = tree_vocabu[words[i]]
    return res

def Root2matrix(NL):
    res = np.zeros([1, parent_len])
    words = NL.split()
    for i in range(len(words)):
        words[i] = words[i] + "_root"
        if words[i] in tree_vocabu:
            res[0, i] = tree_vocabu[words[i]]
    return res


def Root2matrix_var(NL):
    res = np.zeros([1, parent_len])
    words = NL.split()
    for i in range(len(words)):
        words[i] = words[i] + "_root"
        if words[i] in tree_vocabu_var:
            res[0, i] = tree_vocabu_var[words[i]]
    return res


def Root2matrix_func(NL):
    res = np.zeros([1, parent_len])
    words = NL.split()
    for i in range(len(words)):
        words[i] = words[i] + "_root"
        if words[i] in tree_vocabu_func:
            res[0, i] = tree_vocabu_func[words[i]]
    return res

class Javaoutput:
    def __init__(self, Tree, Nl, Node, PNode , Root, TreeWithEnd,FatherTree, GrandFatherTree, state):
        self.Tree = Tree
        self.Nl = Nl
        self.Node = Node
        self.PNode = PNode
        self.Root = Root
        self.Probility = 1
        self.is_end = False
        self.state = state
        self.FuncDict = {}
        self.FuncList = []
        self.VarList = []
        self.RuleList = []
        self.FatherTree = FatherTree
        self.TreeWithEnd = TreeWithEnd
        self.GrandFatherTree = GrandFatherTree
        self.list2wordlistjava = []
        self.numberstack = []

    def prin(self):
        print(self.Tree)

    def __lt__(self, other):
        return self.Probility > other.Probility


def getJavaOut(Nl):
    f = open("Tree_Feature.out", "r")
    lines = f.readlines()
    f.close()
    # print(lines)
    if len(lines) == 2:
        return Javaoutput(lines[0][:-1], Nl, "", "", "", "", "", "", "end")
    if len(lines) == 12:
        return Javaoutput(lines[4][:-1], Nl, lines[1][:-1], lines[2][:-1], lines[3][:-1], lines[0][:-1],lines[6][:-1], lines[7][:-1], "end")

    if len(lines) == 1:
        return Javaoutput("", Nl, "", "", "", "", "", "", "error")
    return Javaoutput(lines[4][:-1], Nl, lines[1][:-1], lines[2][:-1], lines[3][:-1], lines[0][:-1],lines[6][:-1], lines[7][:-1], "grow")


def getnode_gen (JavaOut):
    ret = []
    countall = 0
    WithEnd = str(JavaOut.TreeWithEnd).split()
    for site in range(0,len(WithEnd)):
        if WithEnd[site] in Rule :
            countall += 1
            if WithEnd[site + 1] == "^":
                break
    nodetree = str(JavaOut.Tree).split()
    fatree = str(JavaOut.FatherTree).split()
    gfatree = str(JavaOut.GrandFatherTree).split() 
    if len(fatree) == 0:
         return ret
    ans = "" # str(nodetree[0])
    ans1 = ""#  str(fatree[0])
    ans2 = ""# str(gfatree[0])
    nowcount = 0
    dtusespace = True
    for i in range(0,len(nodetree)):
        if dtusespace:
            ans += nodetree[i]
            ans1 += fatree[i]
            ans2 += gfatree[i]
        else:
            ans += " " + nodetree[i]
            ans1 += " " + fatree[i]
            ans2 += " " + gfatree[i]
        dtusespace = False
        if nodetree[i] in Rule:
            nowcount += 1
            if countall == nowcount:
                nowcount += 1
                ans += " node_gen ^"
                ans1 += " " + nodetree[i] + " ^"
                ans2 += " " + fatree[i] + " ^"
    ret.append(ans)
    ret.append(ans1)
    ret.append(ans2)
    return ret

def getlistDeep_all(inputlist):
    ne = []
    count = 0
    for p in inputlist:
        if p == "^":
            count -= 1
            ne.append(count)
        else:
            ne.append(count)
            count += 1
    return ne

def cov(tree):
    ans = " "
    li = tree.split()
    #for s in str:
    deeplist = getlistDeep_all(li)
    mp = {}
    for i in range(len(li)):
        if li[i] == "^":
            now = deeplist[i]
            li[i] = mp[now] + "^"
        else:
            mp[deeplist[i]] = li[i]
        ans += " " + li[i]
    return ans.replace("  ", "")

def back(treeset):
    ret = []
    for tree in treeset:
        ret.append(cov(tree))
    return ret

def getAction(sess, Code_gen_model, JavaOut):
    input_NL = NL2matrix(JavaOut.Nl)#.replace("(","").replace(")","").replace("#","").replace("$",""))
    treeset = getnode_gen(JavaOut)
    treesetnew = back(treeset)
    print(treeset[0])
    if len(treeset) == 0 :
        JavaOut.is_end = True
        return np.zeros([1,classnum])
    input_Tree = Tree2matrix(treesetnew[0])#.replace("ALL1","ALL").replace("COMMON1","COMMON").replace("TOTEM1","TOTEM"))
    input_PL = Root2matrix(JavaOut.Root)
    input_Node = Tree2matrix(treesetnew[1])
    input_PNode = Tree2matrix(treesetnew[2])
    JavaOutNext = JavaOut
    JavaOutNext.Tree = deepcopy(JavaOut.Tree)
    JavaOutNext.Tree = treeset[0]
    #print (addfeature(JavaOutNext))
    if (len(addfeature(JavaOutNext)) > 10):
        return np.zeros([1, classnum])
    
    FuncList = []
    for word in addfeature(JavaOut):
        FuncList.append(word + "_func")
    
    input_FuncList = Tree2matrix2(FuncList, 1)
    #print (JavaOutNext.FuncDict)
    #print (input_FuncList)
    #print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    res = np.zeros([1,400])
    for i in range(len(JavaOut.RuleList)):
        res[0,i] = JavaOut.RuleList[i]
    input_RL = res
    #print (input_NL)
    #print (input_Tree)
    #print (input_PL)
    #print (input_Node)
    #print (input_PNode)
    #print (input_RL)
    #print (input_FuncList)
#    if JavaOut.Node in tree_vocabu:
#        input_Node[0, 0] = tree_vocabu[JavaOut.Node]
#    input_PNode = np.zeros([1, 1])
#    if JavaOut.PNode in tree_vocabu:
#        input_PNode[0, 0] = tree_vocabu[JavaOut.PNode]
    res = Code_gen_model.y_result.eval(session=sess, feed_dict={
        Code_gen_model.input_NL: input_NL,
        Code_gen_model.input_Tree: input_Tree,
        Code_gen_model.inputN: input_Node,
        Code_gen_model.inputP: input_PNode,
        Code_gen_model.inputparentlist: input_PL,
        Code_gen_model.inputrulelist: input_RL,
        Code_gen_model.inputunderfunclist: input_FuncList,
        Code_gen_model.keep_prob:1,
        Code_gen_model.is_train:False})
    return res

def getAction_var(sess, Code_gen_model, JavaOut):
    input_NL = NL2matrix_var(Ori_Nl)
    treeset = []
    treeset.append(JavaOut.Tree)
    treeset.append(JavaOut.FatherTree)
    treeset.append(JavaOut.GrandFatherTree)
    treeset = back(treeset)
    input_Tree = Tree2matrix_var(treeset[0])
    input_Root = Root2matrix_var(JavaOut.Root)
    VarList = []
    for word in JavaOut.VarList:
        VarList.append(word+"_var")
    input_A = Tree2matrix_var2(VarList,400)
    input_PTree = Tree2matrix_var(treeset[1])
    input_GPTree = Tree2matrix_var(treeset[2])
    FuncList = []
    for word in addfeature(JavaOut):
        FuncList.append(word + "_func")
    input_FuncList = Tree2matrix_var2(FuncList, 1)

    res = Code_gen_model.y_result.eval(session=sess, feed_dict={
        Code_gen_model.input_NL: input_NL,
        Code_gen_model.input_Tree: input_Tree,
        Code_gen_model.inputN: input_PTree,
        Code_gen_model.inputP: input_GPTree,
        Code_gen_model.inputparentlist: input_Root,
        Code_gen_model.inputrulelist: input_A,
        Code_gen_model.inputunderfunclist: input_FuncList,
        Code_gen_model.keep_prob: 1.0,
        Code_gen_model.is_train: False})
    return res

funcnum = 0
nlnum = 0

def getAction_func(sess, Code_gen_model, JavaOut):
    global funcnum
    input_NL = NL2matrix_func(Ori_Nl)
    treeset = []
    treeset.append(JavaOut.Tree)
    treeset.append(JavaOut.FatherTree)
    treeset.append(JavaOut.GrandFatherTree)
    treeset = back(treeset)
    input_Tree = Tree2matrix_func(treeset[0])#.replace("ALL1","ALL").replace("COMMON1","COMMON").replace("TOTEM1","TOTEM"))
    input_Root = Root2matrix_func(JavaOut.Root)
    RuleList = []
    for word in JavaOut.FuncList:
        RuleList.append(word + "_rule")
    input_A = Tree2matrix_func2(RuleList,400)
    input_PTree = Tree2matrix_func(treeset[1])
    input_GPTree = Tree2matrix_func(treeset[2])
    FuncList = []
    for word in addfeature(JavaOut):
        FuncList.append(word + "_func")
    input_FuncList = Tree2matrix_func2(FuncList, 1)

    funcnum += 1

    res = Code_gen_model.y_result.eval(session=sess, feed_dict={
        Code_gen_model.input_NL: input_NL,
        Code_gen_model.input_Tree: input_Tree,
        Code_gen_model.inputN: input_PTree,
        Code_gen_model.inputP: input_GPTree,
        Code_gen_model.inputparentlist: input_Root,
        Code_gen_model.inputrulelist: input_A,
        Code_gen_model.inputunderfunclist: input_FuncList,
        Code_gen_model.keep_prob: 1.0,
        Code_gen_model.is_train: False})
    return res


#Rule = []
Func = []
Var = []
End = []
One = []
gen = []


def ReadRule ():
    f = open("../model/tree/Rule.txt","r")
    lines = f.readlines()
    #print(lines)
    i=0
    for line in lines:
        word = str(line)[:-1].split(" ")
        Rule.append(word[1])
        if "gen_func" in word:
            gen.append(i)
        i += 1
        # if "End" in word :
        #     End.append(i)
        # i += 1

def ReadGenRule():
    f = open("../model/func/Rule.txt","r")
    lines = f.readlines()
    for line in lines:
        string = str(line).replace(" ","")[:-1]
        Func.append(string)

def ReadVarRule():
    f = open("../model/var/Rule.txt","r")
    lines = f.readlines()
    for line in lines:
        string = str(line).replace(" ","")[:-1]
        Var.append(string)



def WriteJavaIn(JavaOut, action):
#    print("------------------------------write----------------------------------")
    f = open("Tree_Rule.in", "w")
    f.write(JavaOut.TreeWithEnd)
#    print(JavaOut.TreeWithEnd)
#    print(str(action))
#    print("-------------------------------End-----------------------------------")
    f.write("\n")
    f.write(str(action))
    f.write("\n")
    f.close()


def isfunc(JavaOut):
    tree = str(JavaOut.Tree).split(" ")
    nowsite = 3
    genis = 0
    for i in range(len(tree)):
        if tree[i] == "gen":
            genis = i - 1
    while (nowsite >= 0 and genis >= 0):
        if tree[genis] == "^":
            nowsite += 1
        else:
            nowsite -= 1
        if nowsite == 0 and tree[genis] == "func":
            return True
        elif nowsite == 0 :
            return False
        genis -= 1
    return False


def addfunc(JavaOut, funcname):
    tree = str(JavaOut.Tree).split(" ")
    nowsite = 4
    genis = 0
    dic = JavaOut.FuncDict
    for i in range(len(tree)):
        if tree[i] == "gen":
            genis = i - 1
    while (nowsite >= 0 and genis >= 0):
        if tree[genis] == "^":
            nowsite += 1
        else:
            nowsite -= 1
        if nowsite == 0 and tree[genis] == "Call":
            dic[genis] = funcname
#            print ("done !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return dic
        elif nowsite == 0 :
            return dic
        genis -= 1
    return dic



def findfather(JavaOut, site):
    tree = str(JavaOut.Tree).split(" ")
    genis = site
    nowsite = 1
    genis -= 1
    while (nowsite >= 0 and genis >= 0):
        if tree[genis] == "^":
            nowsite += 1
        else:
            nowsite -= 1
        if nowsite == 0:# and tree[genis] == "Call":
#            JavaOut.FuncDict[genis] = funcname
            return genis
#        elif nowsite == 0 :
#            return -1
        genis -= 1
    return -1

def addfuncdef(JavaOut, funcname):
    tree = str(JavaOut.Tree).split(" ")
    nowsite = 2
    genis = 0
    dic = JavaOut.FuncDict
#    print (tree)
    for i in range(len(tree)):
        if tree[i] == "gen":
            genis = i - 1
    while (nowsite >= 0 and genis >= 0):
        if tree[genis] == "^":
            nowsite += 1
        else:
            nowsite -= 1
        if nowsite == 0 and tree[genis] == "FunctionDef":
            dic[genis] = funcname
            #print ( JavaOut.FuncDict)
#            print ("done !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return dic
        elif nowsite == 0 :
            return dic
        genis -= 1
    return dic



def addfeature(JavaOut) :
    tree = str(JavaOut.Tree).split(" ")
    genis = 0
    flist = []

    for i in range(len(tree)):
        if tree[i] == "gen_func" or tree[i] == "gen" or tree[i] == "node_gen":
            genis = i
    site = genis 
    while (site > 0):
        if tree[site] == "Call" or tree[site] == "FunctionDef":
            if site in JavaOut.FuncDict.keys():
                flist.append(JavaOut.FuncDict[site])
        site = findfather(JavaOut,site)

    return flist
#    while (nowsite >= 0 and genis >= 0):
#        if tree[genis] == "^":
#            nowsite += 1
#        else:
#            nowsite -= 1
#        if nowsite == 0 and tree[genis] == "Call":
#            JavaOut.FuncDict[genis] = funcname
#            return True
#        elif nowsite == 0 :
#            return False
#        genis -= 1
#    return False




def BeamSearch(sess, sess_var, sess_func, Code_gen_model, gen_var, gen_func, Nl, N, NL_number):
    Javaout = getJavaOut(Nl)
    Javaout.numberstack = deepcopy(numberstack)
    Javaout.list2wordlistjava = deepcopy(list2wordlist)
    close_table = {}
    close_table[Javaout.Tree] = 1
    Beam = [Javaout]
    Set_ = Q.PriorityQueue()
    testsizefunc = 0
    testsizevar = 0
    level = 0
    print("work begin")
    while True:
        print("search level is " + str(level))
        level += 1
        Set_ = Q.PriorityQueue()

        for JavaOut in Beam:
            if JavaOut.is_end :
                Set_.put(JavaOut)
                continue
            res = getAction(sess, Code_gen_model, JavaOut)
            
            #print(JavaOut.Nl)
            #print(JavaOut.Tree)
            #print(JavaOut.Node)
            #print(JavaOut.PNode)
            #print(JavaOut.Root)
            #print(JavaOut.RuleList)
            #print(JavaOut.TreeWithEnd)
            # print(res)
            # print (res)
            sumpro = 0
            maxpro = -1
            listselect = []
            for i in range(int(res.shape[1])):
                #print(i)
                # WriteJavaIn(JavaOut, i)
                if Rule[i] == JavaOut.Node:
                    sumpro += res[0, i]
                    listselect.append(res[0, i])
                    maxpro = max(res[0, i],maxpro)
            listselect = sorted(listselect,reverse=True)
            delete = 0
            if len(listselect)>N:
                delete = listselect[N-1]
#            if maxpro != -1 and maxpro < 0.4 :
#                WriteJavaIn(JavaOut, -1)
#                if os.system("java -jar transfer_ast_jar/transfer_ast.jar") != 0:
#                    print ("error")
#                else:
#                    print ("grow")
#                    JavaOutNext = getJavaOut(Nl)
#                    JavaOutNext.RuleList = deepcopy(JavaOut.RuleList)
#                    JavaOutNext.FuncList = deepcopy(JavaOut.FuncList)
                #JavaOutNext.RuleList = deepcopy(JavaOut.RuleList)
#                    JavaOutNext.VarList = deepcopy(JavaOut.VarList)
#                    JavaOutNext.Probility = JavaOut.Probility * (1 - sumpro)
#                    print(1 - sumpro)
#                    if JavaOutNext.state == "end":
#                        JavaOutNext.is_end = True
#                    Set_.put(JavaOutNext)
#                    continue
            added = 0
            for i in range(int(res.shape[1])):
                # print(i)
                # WriteJavaIn(JavaOut, i)
                if Rule[i] != JavaOut.Node:
                    # print(Rule[i])
                    # print(JavaOut.Node)
                    # print("WrongNode")
                    added += 1
                #    delete = listselect[min(N - 1 + added, len(listselect) - 1)]
                    continue
                if res[0, i] < delete:
                    continue
                # if JavaOut.Node in One and res[0,i]<0.2 and not i in End:
                #     continue
                WriteJavaIn(JavaOut, i )
                GenRoot = JavaOut.Root
                if os.system("java -jar transfer_ast_jar/transfer_ast.jar") != 0:
                    print ("error")
                # print ("Java Run")
                JavaOutNext = getJavaOut(Nl)
                JavaOutNext.FuncList = deepcopy(JavaOut.FuncList)
                JavaOutNext.RuleList = deepcopy(JavaOut.RuleList)
                JavaOutNext.VarList = deepcopy(JavaOut.VarList)
                JavaOutNext.FuncDict = deepcopy(JavaOut.FuncDict)
                JavaOutNext.list2wordlistjava = deepcopy(JavaOut.list2wordlistjava)
                JavaOutNext.numberstack = deepcopy(JavaOut.numberstack)
                nowtree = JavaOutNext.Tree

#    Javaout.numberstack = deepcopy(numberstack)
#    Javaout.list2wordlistjava = deepcopy(list2wordlist)
                if i in gen :# and JavaOutNext.state != "end":
                    rule = i
                    #print("Call Gen")
                    JavaOutNext.Tree = nowtree
                    JavaOutNext.Root = GenRoot
                    spl = str(JavaOutNext.Tree).split()
                    for site in range(len(spl)):
                        if spl[site] == "gen_func":
                            JavaOutNext.Tree = str(JavaOutNext.Tree).replace("gen_func","gen")
                            if isfunc(JavaOutNext):
                                #print("Call func")

                                testsizefunc += 1

                                global nlnum
                                nlnum = NL_number
                                genres = getAction_func(sess_func,gen_func,JavaOutNext)
                         #       print(genres)
                                for i in range(int(genres.shape[1])):
                                    JavaOutNext = getJavaOut(Nl)
                                    JavaOutNext.FuncList = deepcopy(JavaOut.FuncList)
                                    JavaOutNext.RuleList = deepcopy(JavaOut.RuleList)
                                    JavaOutNext.FuncDict = deepcopy(JavaOut.FuncDict)
                                    JavaOutNext.RuleList.append(rule)
                                    JavaOutNext.VarList = deepcopy(JavaOut.VarList)
                                    JavaOutNext.list2wordlistjava = deepcopy(JavaOut.list2wordlistjava)
                                    JavaOutNext.numberstack = deepcopy(JavaOut.numberstack)
                                    JavaOutNext.Tree = str(JavaOutNext.Tree).replace("gen_func","gen")
                                    JavaOutNext.FuncDict = addfunc(JavaOutNext, Func[i])
                                    JavaOutNext.Tree = str(JavaOutNext.Tree).replace("gen",Func[i])
                                    JavaOutNext.TreeWithEnd = str(JavaOutNext.TreeWithEnd).replace("gen_func",Func[i])
#                                    addfunc(JavaOutNext, Func[i])

                                    '''if len(JavaOutNext.FuncList) > 0 :
                                        JavaOutNext.FuncList.append("")'''
                                    if len(JavaOutNext.FuncList) >= 40:
                                        JavaOutNext.is_end = True
                                    else:
                                        JavaOutNext.FuncList.append(Func[i])
                                        JavaOutNext.Probility = JavaOut.Probility * genres[0, i]
                                    if JavaOutNext.state == "end":
                                        JavaOutNext.is_end = True

                                    Set_.put(JavaOutNext)
                            else:
                                #print("Call Var")
                                genres = getAction_var(sess_var,gen_var,JavaOutNext)

                                testsizevar += 1

                        #        print(genres)
                                for i in range(int(genres.shape[1])):
                                    JavaOutNext = getJavaOut(Nl)
                                    JavaOutNext.FuncList = deepcopy(JavaOut.FuncList)
                                    JavaOutNext.RuleList = deepcopy(JavaOut.RuleList)
                                    JavaOutNext.FuncDict = deepcopy(JavaOut.FuncDict)
                                    JavaOutNext.RuleList.append(rule)
                                    JavaOutNext.list2wordlistjava = deepcopy(JavaOut.list2wordlistjava)
                                    JavaOutNext.numberstack = deepcopy(JavaOut.numberstack)
                                    JavaOutNext.VarList = deepcopy(JavaOut.VarList)
                                    what2replace = Var[i]
                                    fa = spl[site-1]
                                    if what2replace == "line:9" and ( fa == "Num" or fa == "num"):
                                        what2replace = getnumberstack(JavaOutNext)
                                    if "line" in what2replace:
                                        what2replace = line2word_var(what2replace,fa,JavaOutNext)

                       #             print(what2replace)
                                    if what2replace == "0" and Var[i] == "line:7":
                                        what2replace = "Nil"
                                    if what2replace == "target":
                                        what2replace = "target0"
                                    if what2replace == "targets" :
                                        what2replace = "targets0"
                                    JavaOutNext.Tree = str(JavaOutNext.Tree).replace("gen_func","gen")
                                    JavaOutNext.FuncDict = addfuncdef(JavaOutNext, what2replace)
                      #              print ( JavaOutNext.FuncDict)
                                    JavaOutNext.Tree = str(JavaOutNext.Tree).replace("gen",what2replace)
                                    JavaOutNext.TreeWithEnd = str(JavaOutNext.TreeWithEnd).replace("gen_func",what2replace)
                      #              print (JavaOutNext.Tree)
                                    if len(JavaOutNext.VarList) >= 100:
                                        JavaOutNext.is_end = True
                                    else:
                                        if Var[i] == "line:9" and not (fa == "Num" or fa == "num"):
                                            if what2replace == "target0":
                                                what2replace = "target"
                                            if what2replace == "targets0" :
                                                what2replace = "targets"
                                            JavaOutNext.VarList.append(what2replace)
                                        else :
                                            JavaOutNext.VarList.append(Var[i])
                                        #print (JavaOutNext.VarList)
                                        JavaOutNext.Probility = JavaOut.Probility * genres[0, i]
                                        if what2replace == "-12345":
                                            JavaOutNext.Probility = 0
                                    if JavaOutNext.state == "end":
                                        JavaOutNext.is_end = True
                                    Set_.put(JavaOutNext)
#                            if JavaOutNext.state == "end":
                                
                elif JavaOutNext.state == "grow":
                    #print("grow")
                    JavaOutNext.Probility = JavaOut.Probility * res[0, i]
                    if len(JavaOutNext.RuleList) >= 300:
                        JavaOutNext.is_end = True
                    else:
                        JavaOutNext.RuleList.append(i)
                    #print(res[0,i])
                    Set_.put(JavaOutNext)

                elif JavaOutNext.state == "end":
                    JavaOutNext.Probility = JavaOut.Probility * res[0, i]
                    JavaOutNext.is_end = True
                    Set_.put(JavaOutNext)
                    # print("end")
                    # f = open("out/" + str(NL_number) + ".txt", "a")
                    # f.write(JavaOutNext.TreeWithEnd)
                    # f.write("\t")
                    # f.write(str(JavaOut.Probility * res[0, i]))
                    # f.write("\n")
                    # f.close()

        Beam = []
        endnum = 0

        #print ("Which followed is Set_")
        #print (Set_)
        while((not Set_.empty()) and N > len(Beam)):
            JavaOut = Set_.get()
            #print(JavaOut.Probility)
            # if JavaOut.Tree not in close_table:
            close_table[JavaOut.Tree] = 1
            Beam.append(JavaOut)
            input_NL = NL2matrix(JavaOut.Nl)
            input_Tree = Tree2matrix(JavaOut.Tree)
            input_PL = Root2matrix(JavaOut.Root)
            input_Node = np.zeros([1, 1])
            res = np.zeros([1,400])
            for i in range(len(JavaOut.RuleList)):
                res[0,i] = JavaOut.RuleList[i]
                input_RL = res
            if JavaOut.Node in tree_vocabu:
                input_Node[0, 0] = tree_vocabu[JavaOut.Node]
                input_PNode = np.zeros([1, 1])
            if JavaOut.PNode in tree_vocabu:
                input_PNode[0, 0] = tree_vocabu[JavaOut.PNode]
            #f.write(str(JavaOut.Node))
            #print(JavaOut.PNode)
            #print(JavaOut.Root)
            #print(JavaOut.RuleList)
           
            #print (JavaOut)
            #print(JavaOut.Nl)
            #print(JavaOut.Tree)
            #print(JavaOut.Node)
            #print(JavaOut.PNode)
            #print(JavaOut.Root)
            #print(JavaOut.RuleList)
            # print(res)
            # print (res)
            if JavaOut.is_end:
                endnum += 1
            if level >= 10000:
                N -= 1

        #if Set_.empty() and level >= 20000:
        #    break

        if endnum >= N:
            # print("end")
            f = open("out/"+str(NL_number)+".txt","w")
            for JavaOut in Beam:
                f.write(JavaOut.Tree)
                f.write("\n")
                f.write(str(JavaOut.Probility))
                f.write("\n")
            f.close()
            break


def predict():
    global Ori_Nl
    
    loadvoc()
    loadvoc_func()
    loadvoc_var()
    #print (len(tree_vocabu_var))
    ReadRule()
    ReadGenRule()
    ReadVarRule()
    gtree = tf.Graph()
    gfunc = tf.Graph()
    gvar = tf.Graph()
    NL_vocabu_size = len(vocabu)
    Tree_vocabu_size = len(tree_vocabu)
    NL_vocabu_size_func = len(vocabu_func)
    Tree_vocabu_size_func = len(tree_vocabu_func)
    NL_vocabu_size_gen = len(vocabu_var)
    Tree_vocabu_size_gen = len(tree_vocabu_var)
    print ("load ")
    with gvar.as_default():
        Code_gen_model_var = gen_var(classnum_gen, embedding_size_gen, conv_layernum_gen, conv_layersize_gen, rnn_layernum_gen,
                                     batch_size_gen, NL_vocabu_size_gen, Tree_vocabu_size_gen, NL_len_gen, Tree_len_gen, parent_len,
                                     learning_rate_gen, keep_prob_gen)

    with gtree.as_default():
        Code_gen_model = code_gen_model(classnum, embedding_size, conv_layernum, conv_layersize, rnn_layernum,
                                        batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, parent_len,
                                        learning_rate, keep_prob)

    with gfunc.as_default():
        Code_gen_model_func = gen_func(classnum_func, embedding_size_func, conv_layernum_func, conv_layersize_func,
                                       rnn_layernum_func, batch_size_func, NL_vocabu_size_func, Tree_vocabu_size_func, NL_len_func,
                                       Tree_len_func, parent_len, learning_rate_func, keep_prob_func)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config = tf.ConfigProto(device_count={"GPU": 0})
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
    #config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    #config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config,graph=gtree) as sess:
        create_model(sess)
        with tf.Session(config = config,graph=gfunc) as sess_func:
            create_model_func(sess_func)
            with tf.Session(config = config,graph=gvar) as sess_var:
                create_model_var(sess_var)
                print ("created model")
                for i in range(1,67):
                    filename = "test/test_input_our_ast_bfs/" + str(i) + ".txt"
                    if not os.path.exists(filename):
                        continue
                    f = open(filename, "r")
#                    Nl = f.readline()[:-1].replace("+5_nl /b_nl","/b_nl").replace("+1_nl /b_nl","/b_nl").replace("spell____nl damage____nl","spelldamage____nl").replace("divine____nl shield____nl","divineshield____nl")
                    Nl = f.readline()[:-1]
                    print(Nl)
                    f.close()
                    filename = "test/test_input_our_ast_bfs/" + str(i) + ".txt"
                    if not os.path.exists(filename):
                        continue
                    f = open(filename, "r")
                    Ori_Nl = f.readline()[:-1]
                    print(Ori_Nl)
                    f.close()
                    readbacklist(i)
                    f = open("Tree_Feature.out", "w")
                    f.write("root ^")
                    f.write("\n")
                    f.write("root")
                    f.write("\n")
                    f.write("Unknown")
                    f.write("\n")
                    f.write("root\n")
                    f.write("root ^\n")
                    f.write("root ^\n")
                    f.write("Unknown ^\n")
                    f.write("Unknown ^\n")
                    f.close()

                    BeamSearch(sess, sess_var ,sess_func, Code_gen_model, Code_gen_model_var, Code_gen_model_func, Nl, 5, i)
                    print(str(i) + "th card is finished")


def main():
    if sys.argv[1] == "eval":
        os.system("cd test/test_bleu && python3 gener.py")
        os.system("cd test/test_bleu && python3 ast2code.py")
    else:
        print ("predict start")
        predict()


main()
