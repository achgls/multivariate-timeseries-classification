import os
import numpy as np

data_dir = "../res"

x = np.load(os.path.join(data_dir, "x_train.npy"))
y = np.load(os.path.join(data_dir, "y_train.npy"))

total_n_samples = x.shape[0]
n_ticks = x.shape[1]
n_dims = x.shape[2]

# Number of samples for each class in the given dataset
class_counts = np.unique(y, return_counts=True)[1]
n_labels = len(class_counts)

# Class weights for loss weighting as a mean to compensate for class imbalance
class_loss_weights = [
    (1 / class_counts[k]) * (total_n_samples / n_labels) for k in range(n_labels)
]

# Initial biases for the output neurons
initial_class_biases = [
    np.log(class_counts[k] / (total_n_samples - class_counts[k])) for k in range(n_labels)
]