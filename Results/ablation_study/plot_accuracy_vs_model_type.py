import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1,"../")
from load_accuracies_all_models import load_accuracies_all_models

if __name__ == "__main__":
    model_name = "UNetCNP"
    model_size = "medium_dropout"
    semantics = True
    percentage_unlabelled_set = 0.25/4
    data_versions = list(range(1,11))
    base_dir = "../../"

    all_accuracies = load_accuracies_all_models(base_dir,model_name,model_size,percentage_unlabelled_set,data_versions,semantics)

    x_ticks_location = [j*3 + 1.5 for j in range(4)]
    x_ticks_label = ["UNet","UNet + CL", "UNet + ET", "UNet + CL + ET"]

    dir_plot = "../figures/write_up/ablation_study/scatter_" + model_name + "_" + model_size \
               + ("_semantics" if semantics else "_") \
               + str(percentage_unlabelled_set) + "P.svg"
    dir_plot_connected = dir_plot[:-4] + "_connected.svg"

    plt.figure()

    x_connected = []
    y_connected = []
    for i,(key,value) in enumerate(all_accuracies.items()):
        x_val = i//2 * 3 + i % 2 + 1
        x_connected.append(x_val)
        y_connected.append(value)

        if "GN" in key:
            style = "r."
            label = "With GradNorm"
        else:
            style = "b."
            label = "Without GradNorm"

        ys = np.array(value)
        xs = np.ones(ys.shape) * x_val
        if i < 2:
            plt.plot(xs,ys,style,label=label)
        else:
            plt.plot(xs, ys, style)
        plt.plot()
    plt.ylabel("Accuracy",fontsize=15)
    plt.xticks(x_ticks_location,x_ticks_label,fontsize=12)
    plt.legend(fontsize="x-large")
    plt.savefig(dir_plot)

    y_connected = np.array(y_connected)
    plt.plot(x_connected,y_connected,"k-",alpha=0.5)

    plt.savefig(dir_plot_connected)




