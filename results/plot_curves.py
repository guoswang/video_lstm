#! /usr/bin/env python
from TensorflowToolbox.utility import file_io
from matplotlib import pyplot as plt
import sys


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage plot_curves file_name.txt")
        exit(1)
    file_name = sys.argv[1]
    file_list = file_io.read_file(file_name)
    label_list = list()
    infer_list = list()
    for f in file_list:
        name, label, infer = f.split(" ")
        label_list.append(float(label))
        infer_list.append(float(infer))

    x = range(len(label_list))

    plt.plot(x, label_list, 'g', x, infer_list, 'r')
    plt.xlim(0, len(label_list))
    plt.show()
