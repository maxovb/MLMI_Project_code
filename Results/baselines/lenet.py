import sys
sys.path.insert(1, "../")
from accuracy_plots import extract_accuracies_from_list_of_files, plot_accuracy

if __name__ == "__main__":
    for dropout in [True,False]:
        acc_dir_plot = "../figures/write_up/baseline/accuracies_lenet" \
                       + ("_dropout" if dropout else "") + ".svg"
        base_dir_txt = "../../saved_models/MNIST/supervised/accuracies/LeNet_" \
                       + ("dropout_" if dropout else "")
        labels = ["Small","Medium","Large"]
        accuracies_dir_txt = [base_dir_txt + x.lower() + ".txt" for x in labels]
        styles_knn = ["k-","k--","k-."]

        accuracies, list_num_samples = extract_accuracies_from_list_of_files(accuracies_dir_txt)
        plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles_knn)
