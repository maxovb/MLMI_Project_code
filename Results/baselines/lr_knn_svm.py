import sys
sys.path.insert(1, "../")
from accuracy_plots import extract_accuracies_from_list_of_files, plot_accuracy

if __name__ == "__main__":
    acc_dir_plot = "../figures/write_up/baseline/accuracies_lr_knn_svm.svg"
    base_dir_txt = "../../saved_models/MNIST/supervised/accuracies/"
    labels = ["LR","SVM","KNN"]
    accuracies_dir_txt = [base_dir_txt + x + ".txt" for x in labels]
    styles_knn = ["k-","k--","k-."]

    accuracies, list_num_samples = extract_accuracies_from_list_of_files(accuracies_dir_txt)
    plot_accuracy(accuracies, list_num_samples, acc_dir_plot, labels, styles=styles_knn)
