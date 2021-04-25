# %%
import matplotlib.pyplot as plt
from math import ceil

def plot_showcase(plot_data, figsize=(18,3), sid=None, gray_list=[]):
    n = len(plot_data)
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, n, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    fig.suptitle(f'Detection Process: Sample #{sid}', fontsize=16)
    for i in range(n):
        img = plot_data[i]
        if i in gray_list:
            axs[0, i].imshow(img, cmap='gray')
        else:
            axs[0, i].imshow(img)
    
    plt.show()
