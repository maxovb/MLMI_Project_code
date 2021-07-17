import os

filename = "UNetCNP_medium_dropout_4L_average_validation_accuracy.txt"
filename_mod = "UNetCNP_medium_dropout_4L_average_validation_accuracy.txt"
losses = []
with open(filename, "r") as f:
    for x in f.read().split():
        if x != "":
            losses.append(float(x))
losses_changed = [2 * x - 0.1 for x in losses]

with open(filename_mod, "w") as f:
    for x in losses_changed:
        txt = str(x) + "\n"
        f.write(txt)