import matplotlib.pyplot as plt
import os

def plot_history(history, directory):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    color = {'training':'blue', 'validation':'red'}
    for phase in history:
        lineloss = axes[0].plot(list(range(len(history[phase]['loss']))), history[phase]['loss'], color=color[phase], label='%s loss'%(phase))
        lineacc = axes[1].plot(list(range(len(history[phase]['accuracy']))), history[phase]['accuracy'], color=color[phase], label='%s accuracy'%(phase))
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='lower right')
    plt.savefig(os.path.join(directory, 'training.svg'), format='svg')

def plot_auc(auc, fpr, tpr, thresholds, directory, name, plot_thresholds):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if plot_thresholds:
        step = int(len(thresholds) / 10)
        for thr in range(0, len(fpr), step):
            plt.text(fpr[thr], tpr[thr], thresholds[thr])
    plt.savefig(os.path.join(directory, name + '_roc.svg'), format='svg')

def plot_accuracy_per_threshold(accuracies, directory, name):
    plt.figure()
    thresholds = sorted(list(accuracies.keys()))
    acc = [accuracies[t] for t in thresholds]
    plt.plot(thresholds, acc)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per threshold')
    plt.savefig(os.path.join(directory, name + '_accuracies.svg'), format='svg')
