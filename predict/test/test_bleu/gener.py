class line2word :
    def __init__(self, f, gf, ggf, nl , ori):
        self.f = f
        self.gf = gf
        self.ggf = ggf
        self.nl = nl
        self.ori = ori
        self.used = False
                                                            

numberstack = []
list2wordlist = []

def readbacklist (Nl_num):
    global numberstack
    global list2wordlist
    numberstack = []
    f = open("../test_output_our_ast_back_past/"+str(Nl_num)+".txt","r")
    lines = f.readlines()
    f.close()
    list2wordlist = []
    for line in lines:
        st = str(line)[:-1].replace("_fu_nc_na_me","").split(" ")
        if len(st)==1:
            numberstack.append(st[0])
        else:
            word = line2word(st[0],st[1],st[2],st[3],st[4])
            list2wordlist.append(word)
                                                                                                            

def line2word_var (line,father,list2word):
    for l in list2word:
        if l.nl == line:
            if l.f == father and l.used == False:
                l.used = True
                return l.ori
            
    for l in list2word:
        if l.nl == line:
            if l.used == False:
                l.used = True
                return l.ori
                
    for l in list2word:
        if l.nl == line:
            l.used = False
           
    for l in list2word:
        if l.nl == line:
            if l.f == father and l.used == False:
                l.used = True
                return l.ori
                
    for l in list2word:
        if l.nl == line:
            if l.used == False:
                l.used = True
                return l.ori            
   
    return "-12345"
                                                                                                                                                                                                                                                                                                    
               
def getnumberstack (numberstack):
    if len(numberstack) == 0 :
        return "-12345"
   
    now = numberstack.pop(0)
    numberstack.append(now)
    return now
    
for i in range(1,67):
    global numberstack
    numberstack = []
    global list2wordlist
    list2wordlist = []
    readbacklist(i)
    f = open("../../out/"+str(i)+".txt","r")
    pro = str(f.readline())
    f.close()
    pro = pro.replace("BootyBayBodyguard","line:0")
    pro = pro.replace("Booty_Bay_Bodyguard","line:0")
    pro = pro.replace("2","line:2")
    pro = pro.replace("3","line:3")
    pro = pro.replace("4","line:4")
    pro = pro.replace("ALL1","line:6")
    pro = pro.replace("TOTEM1","line:7")
    pro = pro.replace("COMMON1","line:8")
    pro = pro.replace("9","line:9")
    pro = pro.replace("1","line:1")
    t = i
    code = pro.split()
    ans = " "
    for i in range(len(code)):
        if "line:" in code[i]:
            st = line2word_var(code[i], code[i - 1], list2wordlist)
            if (code[i-1] == "num" or code[i - 1] == "Num" or code[i-1] == "NUM" or st == "-12345"):
                if code[i] == "line:9":
                    code[i] = getnumberstack(numberstack)
                elif st != "-12345":
                    code[i] = st
            else :
                code[i] = st
        
        ans += " " + code[i]
    i = t
    ans = ans.replace("  ","")
    f = open(str(i) + ".txt", "w")
    f.write(ans)
    f.close()
