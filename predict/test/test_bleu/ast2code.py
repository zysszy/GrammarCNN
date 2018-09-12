import nltk
import os

def findtheson(node):
    output = []
    count = 0
    site = 0
    for i in node[1:]:
        site += 1
        if i == "^":
            count -= 1
        else :
            count += 1
        if count < 0:
            break
        elif count == 1 and i != "^":
            output.append(site)
    return output

def ast2code(ast, d, fa ):
   # print(d)
    son = findtheson(ast)
    #for s in son:
    #    print ("son of " + ast[0] + " : " + ast[s])
    start = ""
    mid = ""
    end = ""
    ans = ""

    if ast[0] == "ClassDef":
       # print(d)
        start =  "class "
        mid = ""
        end = ""

    elif ast[0] == "bases":
        start = "("
        mid = ", "
        end = " ) : "
        if fa == "If":
            start = " "
            end = " : "

    elif ast[0] == "FunctionDef" :

        start = "def "
        mid = " :"
        end = ""

    elif ast[0] == "args" or ast[0] == "keywords":
        start = "("
        mid = ", "
        end = ")"
        if fa == "Lambda" and len(son) <= 1:
            start = ""
            end = "" 

    elif ast[0] == "body" :
       # print(d)
        start =  ""
        mid = "\n" + d * '    '
        end = "\n"

    elif ast[0] == "If" :
        start = "if "
        mid = ""
        end = ""

    elif ast[0] == "For" :
        start = "for "
        mid = " "
        end = ""

    elif ast[0] == "iter":
        start = "in "
        mid = ""
        end = ":"

    elif ast[0] == "While" :
        start = "while "
        mid = ""
        end = ""

    elif ast[0] == "AugAssign":
        mid = "= "

    elif ast[0] == "test" :
        start = "("
        mid = ""
        end = "):"
        if fa == "If":
            start = " "
            end = " : "

    elif ast[0] == "attr" :
        start = "."
        mid = ""
        end = ""

    elif ast[0] == "keyword" :
        start = ""
        mid = " = "
        end = ""

    elif ast[0] == "Return" :
        start = "return "

    elif ast[0] == "orelse" and len(son) > 0 :
        start = "\n" + d * '    ' + "else : "

    elif ast[0] == "List" :
        start = "["
        mid = ", "
        end = "]"

    elif ast[0] == "slice":
        start = "["
        end = "]"

    elif ast[0] == "elts" :
        start = ""
        mid = ", "
        end = ""

    elif ast[0] == "asname":
        start = " as "
        if ast[son[0]]:
            return ""

    elif ast[0] == "ImportFrom" :
        start = "from "
        mid = " import "
        end = ""
        son = son[:-1]

    elif ast[0] == "Assign":
        start = "\n" + d * '    '
        mid = " = "

    elif ast[0] == "Lambda":
        start = " lambda  "
        mid = " : "


    elif ast[0] == "values" and (fa == "or" or fa == "and"):
        mid = " " + fa +" "

    nd = d
    if ast[0] == "body":
        nd = d + 1


    elif ast[0] == "BoolOp":
        ast[0] = ast2code(ast[son[0]:], nd, ast[0])

    if len(son) == 0:
        if ast[0] == "keywords" or ast[0] == "args":
            return "()"
        elif ast[0] == "orelse":
            return ""
        elif ast[0] == "Add" :
            return " +"
        elif ast[0] == "Lt" :
            return " < " 
        elif ast[0] == "LtE" :
            return " <= "
        elif ast[0] == "And" :
            return "and"
        elif ast[0] == "Or" :
            return "or"
        elif ast[0] == "Gt" :
            return " > "
        elif ast[0] == "GtE" :
            return " >= "
        elif ast[0] == "Eq" :
            return " == "
        elif ast[0] == "Is" :
            return " is "
        elif ast[0] == "Not" :
            return " not "
        elif ast[0] == "IsNot" :
            return " is not "
        return ast[0]
    else :
        for i in range(len(son)):
            if i == 0 and (ast[0] == "and" or ast[0] == "or"):
                continue
            if i > 0:
                if ast[0] == "AugAssign" and i == 1:
                    pass
                else :
                    ans += mid
            if ( ast[0] == "keywords" or ast[0] == "body" ) and i == 0:
                ans += mid
            if i + 1 >= len(son):
                ans += ast2code(ast[son[i]:], nd, ast[0])
            else:
                ans += ast2code(ast[son[i]:son[i + 1]], nd, ast[0])

    if fa == "Str" :
        start += "\""
        end = "\"" + end
        ans = ans.replace("_"," ")
    #ans = ans.replace(" p)"," player)")
    return start + ans.replace(")(", "").replace(":(", "(").replace("(,", "(") + end

import re

def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    #code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]

    return tokens

sss = 0

acc = 0
for i in range(1, 67):
    filename = str(i) + ".txt"
    if not os.path.exists(filename):
        continue
    file =  open(str(i) + ".txt","r")
    line = file.readline()
    file.close()
    file = open("ground/" + str(i) + ".txt" , "r")
    li = file.readline()
    file.close()
    print ("-----------------" + str(i) + "---------------------")
    print (ast2code(line.replace(" End ^"," ").replace("_fu_nc_na_me", "").split(), 0, "").replace("target0","target").replace("targets0", "targets"))
    gen = tokenize_for_bleu_eval(ast2code(line.replace(" End ^"," ").replace("_fu_nc_na_me", "").split(), 0, "").replace("target0","target").replace("targets0", "targets"))
    print ("------------------true---------------------")
    print (ast2code(li.replace(" End ^"," ").replace("_fu_nc_na_me", "").split(), 0, ""))
    code = tokenize_for_bleu_eval(ast2code(li.replace(" End ^"," ").replace("_fu_nc_na_me", "").split(), 0, ""))
    print ("------------------bleu---------------------")
    bleu = nltk.translate.bleu_score.sentence_bleu([gen],code)
    if bleu == 1 :
        acc += 1
    sss += bleu
    print (bleu)
    print ("-------------Complete Match----------------")
    print (acc)


print ("-------------------ALL--------------------")
print (sss/66)
