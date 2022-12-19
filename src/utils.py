import os
import numpy as np
import matplotlib.pyplot as plt

# ----- CONSTANTS -----
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
# --------------------

def plot_training_history(history):
    best_epoch = np.argmax(history['val_accuracy'])
    plt.figure(figsize=(17, 4))
    plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Categorical Crossentropy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17, 4))
    plt.plot(history['accuracy'], label='Training accuracy', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_accuracy'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(18, 3))
    plt.plot(history['lr'], label='Learning Rate', alpha=.8, color='#ff7f0e')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()
