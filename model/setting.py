def readrule():
    f = open("Rule.txt", "r")
    lines = f.readlines()
    f.close()
    return len(lines)

NL_len = nl_len = 150  # the number of tokens (input description)
Tree_len = tree_len = 800         # the number of tokens (partial AST)
classnum = readrule()  # the number of rules
parent_len = 100       # Tree Path 
rulelist_len = 400     # Predicted Rules
embedding_size = 128   # embedding dim
conv_layernum = 128    # conv dim
batch_size = 60        # batch size
learning_rate = 1e-4   # learning rate
train_times = 1000     # trainning times
