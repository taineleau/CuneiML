import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data
pred_files = [
    'test1_best_pred_labels.npy',
    'test1_best_pred_labels_vote.npy',
    'test2_best_pred_labels.npy',
    'test2_best_pred_labels_vote.npy',
    'test3_best_pred_labels.npy',
    'test3_best_pred_labels_vote.npy'
]
true_files = [
    'test1_best_true_labels.npy',
    'test1_best_true_labels.npy',
    'test2_best_true_labels.npy',
    'test2_best_true_labels.npy',
    'test3_best_true_labels.npy',
    'test3_best_true_labels.npy'
]

predictions = [np.load(
    f'/graft3/code/tracy/data/predictions/sign/{file}') for file in pred_files]
truths = [np.load(
    f'/graft3/code/tracy/data/predictions/sign/{file}') for file in true_files]

# Plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i in range(6):
    cm = confusion_matrix(truths[i], predictions[i])
    # sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=axes[i])
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='magma', ax=axes[i])

    axes[i].set_title(f'Confusion Matrix {i+1}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('True')

plt.tight_layout()
plt.show()
