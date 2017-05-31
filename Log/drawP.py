import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm



def get_data(file):
    fin = open(file)
    x = []
    y = []
    for line in fin.readlines():
        line = line.strip().split(': ')
        if line[0] == "micro_f1":
            y.append(float(line[1]))
            x.append(len(y))
    return x, y

if __name__ == "__main__":
    
    x, y = get_data("rn0.9.txt")
    plt.plot(x, y)
    plt.legend()
    plt.ylabel("cost")
    plt.xlabel("time(s)")
    #plt.show()
    plt.savefig("picture" + os.sep + "point.png")
