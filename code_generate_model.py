#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.contrib import *

def BatchNormalization(x, phase_train=True, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]),
                                      name='gamma', trainable=True)
        axis = list(range(len(x.get_shape()) - 1))
        batch_mean, batch_var = tf.nn.moments(x, axis, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def matrixMargin(margin, x):
    margin = tf.fill(tf.shape(x), margin)
    result = tf.multiply(margin, x)
    return result

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
        x = self.Relu(x)
        for i in range(stage):
            x = self.MulCnn(x, input_t)
        return x
    
    def MulCnn(self, input_tensor, input_attach):
        x = self.Conv1d(input_tensor, self.conv_layernum, self.conv_layersize, padding='same')
        x = self.Relu(x)
        x = self.Conv1d(x, self.conv_layernum, self.conv_layersize, padding='same')
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

    def max_Attention2(self, state, max_pool):
        state_height = int(state.shape[2])
        pool_height = int(max_pool.shape[1])
        attention_matrix = self.weight_variable(shape=[state_height, pool_height])
        tmp_matrix = tf.einsum("ijk,kl->ijl", state, attention_matrix)
        w_pool = tf.expand_dims(max_pool, -1)
        tmp_matrix = tf.matmul(tmp_matrix, w_pool)
        weight_vec = tf.nn.softmax(tf.reduce_max(tmp_matrix, reduction_indices=[2]))
        weight_vec = tf.expand_dims(weight_vec, -1)
        return weight_vec

    def attention_weight(self, state, max_pool):
        state_height = int(state.shape[1])
        pool_height = int(max_pool.shape[1])
        state = tf.transpose(tf.expand_dims(state, -1), [0, 2, 1])
        attention_matrix = self.weight_variable(shape=[state_height, pool_height])
        tmp_matrix = tf.einsum("ijk,kl->ijl", state, attention_matrix)
        w_pool = tf.expand_dims(max_pool, -1)
        tmp_matrix = tf.matmul(tmp_matrix, w_pool)
        tmp_matrix = tf.nn.sigmoid(tmp_matrix)
        return tmp_matrix

    def Conv2d(self, x, out_channels, kernel_size, padding):
        return tf.layers.conv2d(x, out_channels, kernel_size=[kernel_size, kernel_size], padding=padding)

    def __init__(self, classnum, embedding_size, conv_layernum, conv_layersize, rnn_layernum,
                 batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, learning_rate, keep_prob):
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
        self.layernum = 2 
        self.class_num = classnum
        self.n_stages = 10

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self.input_NL = tf.placeholder(tf.int32, shape=[None, NL_len])
        self.input_Tree = tf.placeholder(tf.int32, shape=[None, Tree_len])
        self.inputY = tf.placeholder(tf.float32, shape=[None, self.class_num])
        self.inputP = tf.placeholder(tf.int32, shape=[None, Tree_len])
        self.inputN = tf.placeholder(tf.int32, shape=[None, Tree_len])
        self.inputR = tf.placeholder(tf.int32, shape=[None, 100])
        self.inputA = tf.placeholder(tf.int32, shape=[None, 100])
        self.inputD = tf.placeholder(tf.int32, shape=[None, 1])
        self.copy = tf.placeholder(tf.float32, shape=[None, classnum, NL_len])
        # embedding
        self.embedding = tf.get_variable("embedding", [NL_vocabu_size ,embedding_size], dtype=tf.float32)
        self.Tree_embedding = tf.get_variable("Tree_embedding", [Tree_vocabu_size, embedding_size], dtype=tf.float32)
        #self.Node_embedding = tf.get_variable("Node_embedding", [Tree_len, embedding_size], dtype=tf.float32)

        em_NL = tf.nn.embedding_lookup(self.embedding, self.input_NL)
        em_Tree = tf.nn.embedding_lookup(self.Tree_embedding, self.input_Tree)
        em_Node = tf.nn.embedding_lookup(self.Tree_embedding, self.inputN)
        em_Parent_Node = tf.nn.embedding_lookup(self.Tree_embedding, self.inputP)
        em_Root = tf.nn.embedding_lookup(self.Tree_embedding, self.inputR)
        em_Api = tf.nn.embedding_lookup(self.Tree_embedding, self.inputA)
        em_Func = tf.nn.embedding_lookup(self.Tree_embedding, self.inputD)
        #weight = matrixMargin(0.02, weight)
        em_stack = tf.stack([em_Tree, em_Node, em_Parent_Node], -2)
        #print(em_stack.shape)
        em_conv = tf.layers.conv2d(em_stack, embedding_size, [1, 3])
        #print(em_conv.shape)
        em_Tree = tf.reduce_max(em_conv, reduction_indices=[-2])
        #embedding attention
        em_Tree = self.Relu(em_Tree)
        #conv
        with tf.variable_scope("Q_conv", reuse=False):
            nl_conv = self.my_conv(em_NL, 10)
        with tf.variable_scope("A_conv", reuse=False):
            tree_conv = self.my_conv(em_Tree, 10)
        with tf.variable_scope("R_conv", reuse=False):
            R_conv = self.my_conv(em_Root, 10)
        with tf.variable_scope("AP_conv", reuse=False):
            AP_conv = self.my_conv(em_Api, 10)
        # max_pool
        nl_one_pool = self.max_height_pooling(nl_one)
        nl_pool = self.max_height_pooling(nl_conv)#tf.reduce_max(Q_conv, reduction_indices=[1])
        tree_pool = self.max_height_pooling(tree_conv)#tf.reduce_max(A_conv, reduction_indices=[1])
        f_pool = self.max_height_pooling(em_Func)
        # attention
        copy_output = self.max_Attention2(nl_conv, tree_pool)
        nl_output = self.max_Attention(nl_conv, f_pool)
        tree_output = self.max_Attention(tree_conv, f_pool)
        root_output = self.max_Attention(R_conv, nl_pool)
        api_output = self.max_Attention(AP_conv, nl_pool)
        self_nl = self.max_Attention(nl_conv, nl_pool)
        func_output = f_pool#self.max_Attention(F_conv, nl_pool)
        
        # connect
        All_q_a = tf.concat([nl_output, tree_output], 1)
        All_q_a = tf.concat([All_q_a, root_output], 1)
        All_q_a = tf.concat([All_q_a, api_output], 1)
        All_q_a = tf.concat([All_q_a, func_output], 1)
        All_q_a = tf.concat([All_q_a, nl_pool], 1)
        All_q_a = tf.concat([All_q_a, tree_pool], 1)
        
        self.fc_layernum = int(All_q_a.shape[1])
        # Fully 
        W_fc = self.weight_variable([int(All_q_a.shape[1]), self.fc_layernum])
        b_fc = self.bias_variable([self.fc_layernum])
        h_fc = tf.nn.tanh(tf.matmul(All_q_a, W_fc) + b_fc)
        # dropout
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)
        # classify
        W_fc2 = self.weight_variable([self.fc_layernum, self.class_num])
        b_fc2 = self.bias_variable([self.class_num])
        self.y_result = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)
        # acc
        W_y_1 = self.attention_weight(nl_output, tree_output)
        W_y_1 = tf.stack([W_y_1, 1 - W_y_1], -1)
        self.y_result = tf.stack([self.y_result, tf.reduce_max(tf.matmul(self.copy, copy_output), reduction_indices=[2])], -1)
        self.y_result = tf.expand_dims(self.y_result, -2)
        self.y_result = tf.reduce_sum(tf.multiply(self.y_result, W_y_1), -1)
        self.y_result = tf.reduce_max(self.y_result, -1)
                                                                


        self.max_res = tf.argmax(self.y_result, 1)
        self.correct_prediction = tf.equal(tf.argmax(self.y_result, 1), tf.argmax(self.inputY, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # cross
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.inputY * tf.log(tf.clip_by_value(self.y_result, 1e-10, 1.0)), reduction_indices=[1]))
        tf.add_to_collection("losses", self.cross_entropy)
        self.loss = tf.add_n(tf.get_collection("losses"))
        # opti
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

