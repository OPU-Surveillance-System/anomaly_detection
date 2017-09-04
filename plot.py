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
