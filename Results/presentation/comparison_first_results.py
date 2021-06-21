from Results.accuracy_plots import  plot_accuracy, extract_accuracies_from_list_of_files, extract_accuracies_form_file_with_multiple_columns

# define the directories
res_lenet = "saved_models/MNIST/supervised_cheat_validation/accuracies/LeNet_large.txt"
res_cnp = "saved_models/MNIST/supervised/accuracies/LR_on_r_CNP_400E.txt"
res_convcnp = "saved_models/MNIST/supervised_semantics/accuracies/LR_on_r_ConvCNP_average_400E.txt"
res_unetcnp = "saved_models/MNIST/supervised_semantics/accuracies/LR_on_r_UNetCNP_average_400E.txt"
res_joint_unetcnp = "saved_models/MNIST/joint_semantics/accuracies/UNetCNP_restrained_LR_4L_average.txt"

# extract the accuracies
# LeNet
acc, list_num_samples = extract_accuracies_from_list_of_files([res_lenet,res_cnp,res_joint_unetcnp])
acc_lenet, acc_cnp, acc_joint_unetcnp = acc[0], acc[1], acc[2]

# ConvCNP
accs, _ = extract_accuracies_form_file_with_multiple_columns(res_convcnp)
acc_convcnp = accs[-1]

# UNetCNP
accs, _ = extract_accuracies_form_file_with_multiple_columns(res_unetcnp)
acc_unetcnp = accs[4]

accuracies = [acc_lenet, acc_cnp, acc_convcnp, acc_unetcnp, acc_joint_unetcnp]

styles = ["r-","b-","g-","k-","k--"]
labels = ["Baseline LeNet", "Fine-tuning CNP", "Fine-tuning ConvCNP", "Fine-tuning UNetCNP", "Joint UNetCNP"]
acc_dir_plot = "Results/presentation/figures/accuracies_first_results_for_presentation.svg"

for i in range(1,len(accuracies)+1):
    acc_dir_plot_local = acc_dir_plot[:-4] + "_" + str(i) + acc_dir_plot[-4:]
    labels_local = labels[:i]
    plot_accuracy(accuracies[:i], list_num_samples, acc_dir_plot_local, labels_local, styles=styles)

