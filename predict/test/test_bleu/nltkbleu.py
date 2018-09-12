import nltk

addt = []
addt.append("If")
addt.append("For")
addt.append("ClassDef")
addt.append("FunctionDef")

def outputt(number):
    st = ""
    for i in range(number):
        # print("\t",end="")
        st += "\t"
    return st

sum = 0 

for i in range(1,67):
    st = []
    try:
        f = open(str(i)+".txt","r")
    except:
        continue
    pro = 1
    line = f.readline()
    tree = str(line).split(" ")
    f.close()
    print("--------------" + str(i) + "---------------")
    numberoft = 0
    inarg = 0
    inkeywords = 0
    outputstr = ""
    infunc = 0
    hasleft = 0
    outputeq = 0
    cardlist = []
    for site in range(len(tree)):
        #print(st)
        if tree[site] == "If":
            cardlist.append("if")
        if tree[site] == "ClassDef":
            cardlist.append("class")
        if tree[site] == "For":
            cardlist.append("for")
        if tree[site] == "FunctionDef":
            cardlist.append("def")
        if tree[site] == "Return":
            cardlist.append("return")

        if tree[site] == "^" and tree[site - 1] != "^":
            if tree[site - 1] != "keywords" and tree[site - 1] != "attr" and tree[site - 1] != "args" and tree[site - 1] != "End" and tree[site - 1] != "None" and tree[site - 1] != "Load" and tree[site - 1] != "kw_defaults" and tree[site - 1] != "kwonlyargs" and tree[site - 1] != "decorator_list" and tree[site - 1] != "defaults":
                cardlist.append(tree[site-1])
                   # cardlist.append("=")
    codelist = cardlist
    
    f = open("../test_output_our_ast/" + str(i) + ".txt","r")
    line = f.readline()
    f.close()
    tree = str(line).replace("_fu_nc_na_me","").split()
    cardlist = []
    for site in range(len(tree)):
        #print(st)
        if tree[site] == "If":
            cardlist.append("if")
        if tree[site] == "ClassDef":
            cardlist.append("class")
        if tree[site] == "For":
            cardlist.append("for")
        if tree[site] == "FunctionDef":
            cardlist.append("def")
        if tree[site] == "Return":
            cardlist.append("return")

        if tree[site] == "^" and tree[site - 1] != "^":
            if tree[site - 1] != "keywords" and tree[site - 1] != "attr" and tree[site - 1] != "args" and tree[site - 1] != "End" and tree[site - 1] != "None" and tree[site - 1] != "Load" and tree[site - 1] != "kw_defaults" and tree[site - 1] != "kwonlyargs" and tree[site - 1] != "decorator_list" and tree[site - 1] != "defaults":
                cardlist.append(tree[site-1])
                   # cardlist.append("=")

    sourcelist = cardlist

    print (sourcelist)
    print (codelist)
    sum += nltk.translate.bleu_score.sentence_bleu([sourcelist],codelist)
    print (nltk.translate.bleu_score.sentence_bleu([sourcelist],codelist))

print ("all-----------------------------")
print (sum/66)
