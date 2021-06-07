import numpy as np
import os

def modify_file(filename):
    lines = []
    with open(filename,'r') as f:
        for line in f:
            lines.append(line)
    if lines[0][:2] == "10":
        return
    else:
        assert lines[0].startswith("training sample sizes:"), "Incorrect file format for file " + filename
        samples = lines[0].split(": ")[1].split()

        with open(filename,'w') as f:
            for i,sample in enumerate(samples):
                txt = sample
                txt += ", " + lines[i+1]
                f.write(txt)

if __name__ == "__main__":
    for dirname in os.listdir("saved_models/MNIST/"):
        if dirname.startswith("supervised"):
            print(dirname)
            for filename in os.listdir("saved_models/MNIST/" + dirname + "/accuracies/"):
                print("saved_models/MNIST/" + dirname + "/accuracies/" + filename)
                modify_file("saved_models/MNIST/" + dirname + "/accuracies/" + filename)
