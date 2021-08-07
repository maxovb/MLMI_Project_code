import sys
sys.path.insert(1, "../")
from accuracy_plots import extract_accuracies_from_list_of_files, plot_accuracy

if __name__ == "__main__":
    for dropout in [True,False]:
        acc_dir_plot = "../figures/write_up/baseline/accuracies_WideResNet.svg"
        base_dir_txt1 = "../../saved_models/MNIST/supervised/accuracies/WideResNet_small.txt"
        base_dir_txt2 = "../../saved_models/MNIST/supervised/accuracies/WideResNet_pretrained_small.txt"
        labels = ["Random initialization","Pre-trained"]
        accuracies_dir_txt = [base_dir_txt1,base_dir_txt2]
        styles_knn = ["k-","k--"]

        accuracies, list_num_samples = extract_accuracies_from_list_of_files(accuracies_dir_txt)
        plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles_knn)
