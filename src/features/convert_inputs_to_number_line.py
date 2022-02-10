"""The purpose of this script is to convert the numerical inputs generated by the generate_data_folder
script into number line images."""
import click
import matplotlib.pyplot as plt
from codetiming import Timer
from humanfriendly import format_timespan

from src.features.number_line import plot_couple_on_line


@click.command()
@click.option('--task_name', type=str)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(task_name):
    with open(f"../data/interim/discrimination_task/{task_name}/inputs.txt", "r") as data_file:
        for line in data_file:
            couple = [float(el) for el in line.split()]
            fig, ax = plot_couple_on_line(couple)
            img_name = "_".join(("discriminate",
                                 str(couple[0]).replace(".", ""),
                                 str(couple[1]).replace(".", "")))
            fig.savefig(f"../data/processed/discrimination_task/{task_name}/{img_name}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
