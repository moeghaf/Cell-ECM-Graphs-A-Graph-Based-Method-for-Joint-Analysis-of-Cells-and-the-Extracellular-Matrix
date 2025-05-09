import matplotlib.pyplot as plt
import numpy as np

def plot_36_images(images, titles=None, cmap='jet', figsize=(10, 10), save_path=None):
    
    assert len(images) == 36, "You must provide exactly 36 images."

    fig, axs = plt.subplots(6, 6, figsize=figsize, dpi=300)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i], cmap=cmap)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i], fontsize=6, pad=2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
