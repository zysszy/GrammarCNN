#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.contrib import *
from setting import *

class code_gen_model:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        weight = tf.Variable(initial)
        tf.add_to_collection("losses", layers.l2_regularizer(0.00005)(weight))
        return weight

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def max_height_pooling(self, input):
        height = int(input.get_shape()[1])
        width = int(input.get_shape()[2])
        input = tf.expand_dims(input, -1)
        output = tf.nn.max_pool(input, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    def max_width_pooling(self, input):
        height = int(input.get_shape()[1])
        width = int(input.get_shape()[2])
        input = tf.expand_dims(input, -1)
        output = tf.nn.max_pool(input, ksize=[1, 1, width, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, height])
        return output

    def my_conv(self, input_t, stage):
        height = int(input_t.get_shape()[1])
        width = int(input_t.get_shape()[2])
        x = self.Conv1d(input_t, self.conv_layernum, self.conv_layersize, padding='same')
        #x = self.BatchNormalization(x, axis=-1, training=self.is_train)
        x = self.Relu(x)
        for i in range(stage):
            x = self.MulCnn(x)
        return x
    
    def MulCnn(self, input_tensor):
        x = self.Conv1d(input_tensor, self.conv_layernum, self.conv_layersize, padding='same')
        #x = self.BatchNormalization(x, axis=-1, training=self.is_train)
        x = self.Relu(x)
        x = self.Conv1d(x, self.conv_layernum, self.conv_layersize, padding='same')
        #x = self.BatchNormalization(x, axis=-1, training=self.is_train)
        x = tf.add_n([x, input_tensor])#x = x + input_tensor
        x = self.Relu(x)
        return x

    def max_Attention(self, state, max_pool):
        state_height = int(state.shape[2])
        pool_height = int(max_pool.shape[1])
        attention_matrix = self.weight_variable(shape=[state_height, pool_height])
        tmp_matrix = tf.einsum("ijk,kl->ijl", state, attention_matrix)
        w_pool = tf.expand_dims(max_pool, -1)
        tmp_matrix = tf.matmul(tmp_matrix, w_pool)
        weight_vec = tf.nn.softmax(tf.reduce_max(tmp_matrix, reduction_indices=[2]))
        weight_vec = tf.expand_dims(weight_vec, -1)
        Out = tf.matmul(state, weight_vec, transpose_a=True)
        out = tf.reduce_max(Out, reduction_indices=[2])
        return out
    
    def __init__(self, classnum, embedding_size, conv_layernum, conv_layersize, rnn_layernum,
                 batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, parent_len, learning_rate, keep_prob):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.vocabu_size = NL_vocabu_size
        self.NL_len = NL_len
        self.Tree_len = Tree_len
        self.conv_layernum = conv_layernum
        self.conv_layersize = conv_layersize
        self.learning_rate = learning_rate
        self.BatchNormalization = tf.layers.batch_normalization
        self.Relu = tf.nn.relu
        self.Conv1d = tf.layers.conv1d
        #self.keep_prob = keep_prob
        self.rnn_layernum = rnn_layernum
        self.layernum = 3
        self.layerparentlist = 3
        self.class_num = classnum
        self.n_stages = 5

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self.input_NL = tf.placeholder(tf.int32, shape=[None, NL_len])
        self.input_Tree = tf.placeholder(tf.int32, shape=[None, Tree_len])
        self.inputY = tf.placeholder(tf.float32, shape=[None, self.class_num])
        self.inputP = tf.placeholder(tf.int32, shape=[None, Tree_len])
        self.inputN = tf.placeholder(tf.int32, shape=[None, Tree_len])
        self.inputparentlist = tf.placeholder(tf.int32, shape = [None, parent_len])
        self.inputrulelist = tf.placeholder(tf.int32, shape = [None, rulelist_len])
        self.inputunderfunclist = tf.placeholder(tf.int32, shape=[None,1])
        self.copy = tf.placeholder(tf.float32, shape=[None, classnum, NL_len])
        self.embedding = tf.get_variable("embedding", [NL_vocabu_size ,embedding_size], dtype=tf.float32)
        self.Tree_embedding = tf.get_variable("Tree_embedding", [Tree_vocabu_size, embedding_size], dtype=tf.float32)
        self.Rule_embedding = tf.get_variable("Rule_embedding", [classnum + 10, embedding_size], dtype=tf.float32)
        em_NL = tf.nn.embedding_lookup(self.embedding, self.input_NL)
        em_Tree = tf.nn.embedding_lookup(self.Tree_embedding, self.input_Tree)
        em_Node = tf.nn.embedding_lookup(self.Tree_embedding, self.inputN)
        em_Parent_Node = tf.nn.embedding_lookup(self.Tree_embedding, self.inputP)
        em_Parent_List = tf.nn.embedding_lookup(self.Tree_embedding, self.inputparentlist)
        em_Rule_List = tf.nn.embedding_lookup(self.Rule_embedding, self.inputrulelist)
        em_Func_List = tf.nn.embedding_lookup(self.Tree_embedding, self.inputunderfunclist)
        # em_Tree = tf.concat([em_Tree, em_Node], 1)
        # em_Tree = tf.concat([em_Tree, em_Parent_Node], 1)
        # em_Tree = tf.concat([em_Tree, self.inputDeep], 1)
        #weight = matrixMargin(0.02, weight) 
        em_stack = tf.stack([em_Tree, em_Node, em_Parent_Node], -2)
        #print(em_stack.shape)
        em_conv = tf.layers.conv2d(em_stack, embedding_size, [1, 3])
        #print(em_conv.shape)
        em_Tree = tf.reduce_max(em_conv, reduction_indices=[-2])
        em_Tree = self.Relu(em_Tree)
        #print(em_Tree.shape)
        with tf.variable_scope("Q_conv", reuse=False):
            nl_conv = self.my_conv(em_NL, 10)
        with tf.variable_scope("A_conv", reuse=False):
            tree_conv = self.my_conv(em_Tree, 10)
        with tf.variable_scope("PL_conv", reuse=False):
            R_conv = self.my_conv(em_Parent_List, 10)
        with tf.variable_scope("FL_conv", reuse=False):
            F_conv = em_Func_List
        with tf.variable_scope("RL_conv", reuse=False):
            RL_conv = self.my_conv(em_Rule_List, 10)

        # max_pool
        nl_pool = self.max_height_pooling(nl_conv)
        tree_pool = self.max_height_pooling(tree_conv)
        pl_pool = self.max_height_pooling(R_conv)
        rl_pool = self.max_height_pooling(RL_conv)
        fl_pool = self.max_height_pooling(F_conv)
        #attention
        nl_output = self.max_Attention(nl_conv, fl_pool)
        tree_output = self.max_Attention(tree_conv, fl_pool)
        root_output = self.max_Attention(R_conv, nl_pool)
        api_output = self.max_Attention(RL_conv, nl_pool)
        func_output = fl_pool
        # connect
        #print("nl", nl_output.shape)
        #self.attention_weight(nl_output, tree_output)
        All_q_a = tf.concat([nl_output, tree_output], 1)
        All_q_a = tf.concat([All_q_a, root_output], 1)
        All_q_a = tf.concat([All_q_a, api_output], 1)
        All_q_a = tf.concat([All_q_a, func_output], 1)
        All_q_a = tf.concat([All_q_a, nl_pool], 1)
        All_q_a = tf.concat([All_q_a, tree_pool], 1)
        #All_q_a = tf.concat([All_q_a, self_nl], 1)
        #All_q_a = tf.concat([All_q_a, nl_one_pool], 1)

        self.fc_layernum = int(All_q_a.shape[1])
        W_fc = self.weight_variable([int(All_q_a.shape[1]), self.fc_layernum])
        b_fc = self.bias_variable([self.fc_layernum])
        h_fc = tf.nn.tanh(tf.matmul(All_q_a, W_fc) + b_fc)
        # dropout
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
        W_fc2 = self.weight_variable([self.fc_layernum, self.class_num])
        b_fc2 = self.bias_variable([self.class_num])
        self.y_result = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)
        self.max_res = tf.argmax(self.y_result, 1)
        self.correct_prediction = tf.equal(tf.argmax(self.y_result, 1), tf.argmax(self.inputY, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.inputY * tf.log(tf.clip_by_value(self.y_result, 1e-10, 1.0)), reduction_indices=[1]))
        tf.add_to_collection("losses", self.cross_entropy)
        self.loss = self.cross_entropy#tf.add_n(tf.get_collection("losses"))
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
#c = code_gen_model(128, 128, 128, 3, 50, 100, 8000, 80000, 150, 800, 8000, 1e-4, 0.5)
