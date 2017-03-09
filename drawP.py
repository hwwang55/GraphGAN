import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm



def get_data(file):
    fin = open(file)
    for i in range(3):  
        fin.readline()
    x = []
    y = []
    for line in fin.readlines():
        line = line.strip().split(' ')
        if len(line) < 3:
            break
        x.append(float(line[2]))
        line = line[3].split('[')[1].split(',')[0]
        y.append(float(line))
    return x, y

if __name__ == "__main__":
    
    cwd = os.getcwd() + os.sep + 'Log'
    for rt, dirs, files in os.walk(cwd):
        cmap = cm.rainbow(np.linspace(0, 1, len(files)))
        for i in range(len(files)):
            x, y = get_data(cwd + os.sep + files[i])
            plt.plot(x, y, color = cmap[i], label = files[i][0:-8])
    plt.legend()
    plt.ylabel("cost")
    plt.xlabel("time(s)")
    #plt.show()
    plt.savefig("picture" + os.sep + "TimeCompare.png")