import matplotlib.pyplot as plt
import numpy as np


def plot_data(data: np.ndarray, rows: int = 5, cols: int = 4, plot_border: bool = True, title: str = "") -> None:
    """Plot the given image data.

    Args:
        data: image data shaped (n_samples, channels, width, height).
        rows: number of rows in the plot .
        cols: number of columns in the plot.
        plot_border: add a border to the plot of each individual digit.
                     If True, also disable the ticks on the axes of each image.
        title: add a title to the plot.

    Returns:
        None

    Note:

    """
    # START TODO ################
    # useful functions: plt.subplots, plt.suptitle, plt.imshow
    # raise NotImplementedError
    # n_samples, channels, width, height = data.shape
    # n_subplots = min(rows * cols, n_samples)
    # fig, axes = plt.subplots(rows, cols)

    # if n_subplots == 1:
    #     axes = np.array([axes])

    # for i in range(n_subplots):
    #     ax = axes[i // cols, i % cols]
    #     imagedata = data[i]
    #     if plot_border:
    #         ax.imshow(np.ones((width + 2, height + 2, channels)))
    #         ax.imshow(imagedata.transpose(1, 2, 0), extent=(1, width, 1, height), origin='upper')
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #     else:
    #         ax.imshow(imagedata.transpose(1, 2, 0))
    # plt.suptitle(title)
    # plt.show()

    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1).axis('on' if plot_border else 'off')
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.imshow(data[i, 0, :, :], cmap='Greys', vmin=0, vmax=1)
    plt.suptitle(title)
    plt.show()
    # END TODO ################
