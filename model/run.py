#-*-coding:utf-8-*-
import sys
# sys.path.append('..')
from readcard import *
from code_generate_model import *
from resolve_data import *
#from model.python_gen.code_generate_model import code_gen_model as gen_func
#from model.python_var.code_generate_model import code_gen_model as gen_var
import os
import tensorflow as tf
import numpy as np
import os
import queue as Q
from copy import deepcopy

#os.envireni
os.environ["CUDA_VISIBLE_DEVICES"]="0"

NL_vocabu_size = len(vocabulary)
Tree_vocabu_size = len(tree_vocabulary)

cardnum = []

card_number = 0

def get_card(lst):
    global cardnum
    if len(cardnum) == 0:
        f = open("nlnum.txt", "r")
        st = f.read()
        cardnum = eval(st)
    dic = {}
    for i, x in enumerate(lst):
        if x == False:
          if cardnum[i] not in dic:
            dic[cardnum[i]] = 1
    return card_number - len(dic)

def create_model(session):
    if(os.path.exists("./" + sys.argv[3] + "/checkpoint")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint("./" + sys.argv[3]))
        print("load the model")
    else:
        session.run(tf.global_variables_initializer())
        print("create a new model")


def save_model(session, number):
    saver = tf.train.Saver()
    saver.save(session,  sys.argv[3] + "/model.cpkt")


def test():
    Code_gen_model = code_gen_model(classnum, embedding_size, conv_layernum, 2, 50,
                  batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, parent_len, learning_rate, 0.5)
    valid_batch = get_valid_batch()
    config = tf.ConfigProto(device_count={"GPU": 0})
    f1 = open("outp.txt", "w")
    f2 = open("numout.txt", "w")
    with tf.Session(config=config) as sess:
      create_model(sess)
      for k in range(len(valid_batch[0])):
          """ac1 = Code_gen_model.correct_prediction.eval(session=sess,
                                        feed_dict={Code_gen_model.input_NL: valid_batch[0][k],
                                                   Code_gen_model.input_Tree: valid_batch[1][k],
                                                   Code_gen_model.inputY: valid_batch[4][k],
                                                   Code_gen_model.inputN: valid_batch[2][k],
                                                   Code_gen_model.inputP: valid_batch[3][k],
                                                   Code_gen_model.inputparentlist: valid_batch[5][k],
                                                   Code_gen_model.inputrulelist:valid_batch[6][k],
                                                   Code_gen_model.inputunderfunclist:valid_batch[7][k][:,:1],	
                                                   Code_gen_model.keep_prob: 1.0,
                                                   Code_gen_model.is_train: False
                                                   })"""
          ac1, numout = sess.run([Code_gen_model.correct_prediction, Code_gen_model.max_res],
                               feed_dict={Code_gen_model.input_NL: valid_batch[0][k],
                                      Code_gen_model.input_Tree: valid_batch[1][k],
                                      Code_gen_model.inputY: valid_batch[4][k],
                                      Code_gen_model.inputN: valid_batch[2][k],
                                      Code_gen_model.inputP: valid_batch[3][k],
                                      Code_gen_model.inputparentlist: valid_batch[5][k],
                                      Code_gen_model.inputrulelist:valid_batch[6][k],
                                      Code_gen_model.inputunderfunclist:valid_batch[7][k][:,:1],
                                      Code_gen_model.keep_prob: 1.0,
                                      Code_gen_model.is_train: False
                                      })
          f1.write(str(ac1))
          f1.write("\n")
          f2.write(str(numout))
          f2.write("\n")
            

def run():
    Code_gen_model = code_gen_model(classnum, embedding_size, conv_layernum, 2, 50,
                                    batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, parent_len, learning_rate, 0.5)
    valid_batch = get_valid_batch()
    best_accuracy = 0.965
    best_card = -1
    #print(NL_vocabu_size, Tree_vocabu_size)
    #config = tf.ConfigProto(device_count={"GPU": 0})
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    f = open("out.txt", "w")
    with tf.Session(config=config) as sess:
        create_model(sess)
        for i in range(train_times):
            batch = get_train_batch(batch_size)
            for j in range(len(batch[0])):
                if j % 400 == 0:
                    print("train " + str(i) + " echos " + str(j) + " times")
                if j % 500 == 0:
                    ac = 0
                    res = []
                    sumac = 0
                    length = 0
                    for k in range(len(valid_batch[0])):
                        '''ac1 = Code_gen_model.accuracy.eval(session=sess,
                                                      feed_dict={Code_gen_model.input_NL: valid_batch[0][k],
                                                                 Code_gen_model.input_Tree: valid_batch[1][k],
                                                                 Code_gen_model.inputY: valid_batch[4][k],
                                                                 Code_gen_model.inputN: valid_batch[2][k],
                                                                 Code_gen_model.inputP: valid_batch[3][k],
                                                                 Code_gen_model.inputparentlist: valid_batch[5][k],
                                                                 Code_gen_model.inputrulelist:valid_batch[6][k],
                                                                 Code_gen_model.inputunderfunclist:valid_batch[7][k][:,:1],	
                                                                 Code_gen_model.keep_prob: 1.0,
                                                                 Code_gen_model.is_train: False
                                                                 })
                        sumac += ac1 * (len(valid_batch[0][k]))
                        length += len(valid_batch[0][k])'''
                        ac1, loss1 = sess.run([Code_gen_model.accuracy, Code_gen_model.correct_prediction],
                                                    feed_dict={Code_gen_model.input_NL: valid_batch[0][k],
                                                               Code_gen_model.input_Tree: valid_batch[1][k],
                                                               Code_gen_model.inputY: valid_batch[4][k],
                                                               Code_gen_model.inputN: valid_batch[2][k],
                                                               Code_gen_model.inputP: valid_batch[3][k],
                                                               Code_gen_model.inputparentlist: valid_batch[5][k],
                                                               Code_gen_model.inputrulelist:valid_batch[6][k],
                                                               Code_gen_model.inputunderfunclist:valid_batch[7][k][:,:1],
                                                               Code_gen_model.keep_prob: 1.0,
                                                               Code_gen_model.is_train: False
                                                               })
                        res.extend(loss1.tolist())
                        ac += ac1;
                    #ac = sumac / length
                    ac /= len(valid_batch[0])
                    card = get_card(res)
                    #loss = 0.0
                    strs = str(ac) + " " + str(card) + "\n"
                    f.write(strs)
                    f.flush()
                    print("current accuracy " +
                          str(ac) + " card is " + str(card))
                    if card > best_card:
                        best_card = card
                        best_accuracy = ac
                        save_model(sess, 1)
                        print("find the better accuracy " +
                              str(best_accuracy) + "in echos " + str(i))
                    elif card == best_card:
                        if(ac > best_accuracy):
                            best_card = card
                            best_accuracy = ac
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
                                                                  Code_gen_model.inputunderfunclist:batch[7][j][:,:1],
                                                                  Code_gen_model.keep_prob: 0.5,
                                                                  Code_gen_model.is_train: True
                                                                  })
        # print(eval(tmpy, "data/q_a_test"))
    f.close()
    print("training finish")
    return


def main():
    global NL_vocabu_size
    global Tree_vocabu_size
    np.set_printoptions(threshold=np.nan)
    # ReadRule()
    if sys.argv[1] == "train":
        print ("detar data")
        os.system("tar -zxvf data_" + sys.argv[3] + ".tar.gz")
        global classnum
        global card_number
        card_number = loadcardnum()
        #os.system("python3 readcard.py train " + sys.argv[2])
        classnum = readrule()
        print ("eval set: " + sys.argv[2])
        print ("loading data ......")
        resolve_data()
        NL_vocabu_size = len(vocabulary)
        Tree_vocabu_size = len(tree_vocabulary)
        print ("finish!")
        run()
    elif sys.argv[1] == "test":
        test()
    else:
        print ("error")
       # predict()


main()
