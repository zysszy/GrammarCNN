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

id = int(input("input the line you want to watch:"))

for i in range(1,67):
    st = []
    try:
        f = open(str(i)+".txt","r")
    except:
        continue
    pro = 1
    for j in range(1, id):
        line = f.readline()
        line = f.readline()
        pro = float(line)
    line = f.readline()
    tree = str(line).split(" ")
    line = f.readline()
    pro = float(line) - pro
    f.close()
    print("--------------" + str(i) + "---------------")
    numberoft = 0
    inarg = 0
    inkeywords = 0
    outputstr = ""
    infunc = 0
    hasleft = 0
    outputeq = 0
    for site in range(len(tree)):
        #print(st)
        if tree[site] == "If":
            outputstr += "\n"+outputt(numberoft) + "if "
            # print("")
            # outputt(numberoft)
            # print("class", end=" ")
        if tree[site] == "ClassDef":
            outputstr += "\n"+outputt(numberoft) + "class "
            # print("")
            outputt(numberoft)
            # print("class", end=" ")
        if tree[site] == "For":
            # print("")
            outputt(numberoft)
            # print("def", end=" ")
            outputstr += "\n"+outputt(numberoft) + "for "
        if tree[site] == "FunctionDef":
            # print("")
            outputt(numberoft)
            # print("def", end=" ")
            outputstr += "\n"+outputt(numberoft) + "def "
        if tree[site] == "body":
            # print("")
            outputt(numberoft)
            outputstr += "\n"+outputt(numberoft)
        if tree[site] == "Return":
            # print("return",end=" ")
            outputstr += "return "

        if tree[site] == "args" or tree[site] == "keywords":
            # print("(",end="")
            inarg += 1

        # if tree[site] == "func":
        #     # print("(",end="")
        #     infunc += 1

        if tree[site] == "keywords":
            # print("(",end="")
            inkeywords += 1
        if tree[site] == "attr" and tree[site + 1] != "End":
            # print(".",end="")
            outputstr += "."
        if tree[site] == "End":
            if tree[site - 1] == "args":
                outputstr += "()"
        if tree[site] == "^" and tree[site - 1] != "^":
            if tree[site - 1] == "args":
                outputstr += "()"
            if tree[site - 1] != "keywords" and tree[site - 1] != "attr" and tree[site - 1] != "args" and tree[site - 1] != "End" and tree[site - 1] != "None" and tree[site - 1] != "Load" and tree[site - 1] != "kw_defaults" and tree[site - 1] != "kwonlyargs" and tree[site - 1] != "decorator_list" and tree[site - 1] != "defaults":
                 # print(tree[site-1],end=" ")
                if inarg > infunc + hasleft :
                    outputstr += "("
                    hasleft += 1
                outputstr += tree[site-1] + " "
                if inarg > 0 :
                    # print(",",end="")
                    outputstr += ","
                if inkeywords > 0:
                    outputeq += 1
                    if outputeq % 2 == 1:
                        outputstr += "= "

        if tree[site] == "^":
            now = st.pop()
            if now in addt:
                numberoft -= 1
            if now == "args" or now == "keywords":
                # print(")",end="")
                inarg -= 1
                if hasleft != 0:
                    outputstr += ")"
                    hasleft -= 1
            # if now == "func":
            #     # print(")",end="")
            #     infunc -= 1
            if now == "keywords":
                # print("(",end="")
                inkeywords -= 1
                outputeq = 0

        else:
            st.append(tree[site])
            if tree[site] in addt:
                numberoft += 1

    print (pro)
    print(outputstr.replace("Store","in").replace(" ,.",".").replace(",)",")").replace(")(",",").replace(",=","=").replace("_fu_nc_na_me","").replace(" ,(","(").replace(" = (","(").replace("= )",")"))
