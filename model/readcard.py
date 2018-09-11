import sys

#card_number = 0

def loadcardnum():
    print ("load card...")

    file = open (sys.argv[2] + ".txt", "r")
    lines = file.readlines()
    file.close()
    file = open("nlnum.txt", "w")
    count = 0
    l = []
    for i in range(len(lines)):
        if i % 8 == 5:
            #if lines[i].strip() == "0":
            if len(lines[i].strip()) == 0:#i >= 8 and lines[i - 8].strip() != lines[i].strip():
                count += 1
            l.append(count)
    file.write(str(l))
    file.close()
    return count
    #global card_number
    #card_number = count
