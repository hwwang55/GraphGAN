import os
import matplotlib as plt
cwd = os.getcwd()



for rt, dirs, files in os.walk(root):
    for file in files:
        x, y = get_data(file)
        plt.plot(x, y)

plt.savefig("TimeCompar.jpg")