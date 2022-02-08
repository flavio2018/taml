"""This file contains functions that create visual representations of numbers in the space, in the form of a number
 line."""
import matplotlib.pyplot as plt


def plot_number_on_line(x, fig, ax, marker="o"):
    ax.scatter(x, 1, color="black", marker=marker)
    fig.tight_layout()
    fig.savefig("../data/interim/number_line/test.png")


def set_background(show_line=True):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(1, 1, figsize=(500*px, 50*px))

    ax.set_xlim(0, 20)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    if show_line:
        ax.axhline(1, color="black")

    return fig, ax


if __name__ == '__main__':
    fig, ax = set_background()
    for marker, position in zip(["o", "s"], [3, 4]):
        plot_number_on_line(position, fig, ax, marker)
