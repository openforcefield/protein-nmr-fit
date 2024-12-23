import json
from pathlib import Path

import click
import numpy
import pandas
import seaborn
from matplotlib import pyplot


def _plot_histogram(
    plot_data_df,
    row_labels: list[str],
    output_path: str,
    histogram_df_column: str,
    figure_size: tuple[float, float],
    bin_width: float = 0.1,
    row_df_column="Replica",
    category_labels: list[str] = ["All"],
    category_df_column: str = None,
    x_label: str = "Collective variable",
    y_label: str = "Fraction",
    x_ticks: list[float] = None,
    y_ticks: list[float] = None,
    x_range: tuple[float, float] = None,
    y_range: tuple[float, float] = None,
):
    figure, axes = pyplot.subplots(
        len(row_labels),
        1,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    min_x = numpy.floor(plot_data_df[histogram_df_column].min() / bin_width) * bin_width
    max_x = numpy.ceil(plot_data_df[histogram_df_column].max() / bin_width) * bin_width
    N_bins = int(numpy.round((max_x - min_x) / bin_width))

    for row_index, row_label in enumerate(row_labels):
        ax = axes[row_index, 0]

        for category_index, category_label in enumerate(category_labels):
            if category_df_column is None:
                category_df = plot_data_df[plot_data_df[row_df_column] == row_label]

            else:
                category_df = plot_data_df[
                    (plot_data_df[row_df_column] == row_label)
                    & (plot_data_df[category_df_column] == category_label)
                ]

            if category_df.shape[0] == 0:
                continue

            histogram_data = category_df[histogram_df_column].values

            if histogram_data.size == 0:
                continue

            histogram_values, histogram_bins = numpy.histogram(
                histogram_data,
                bins=N_bins,
                range=(min_x, max_x),
            )

            # ax.stairs(
            ax.plot(
                (histogram_bins[:-1] + histogram_bins[1:]) / 2,
                histogram_values / histogram_data.size,
                color=seaborn.color_palette()[category_index % 10],
            )

    if x_ticks is None:
        x_ticks = numpy.round(
            numpy.arange(
                numpy.floor(min_x / 30) * 30,
                numpy.ceil(max_x / 30) * 30 + 15,
                30,
            ),
            1,
        )

    if x_range is None:
        x_range = (min_x, max_x)

    if y_ticks is None:
        y_ticks = numpy.arange(0, 1.1, 0.2)

    if y_range is None:
        y_range = (0, 1)

    for ax in axes.flat:
        ax.set_xticks(x_ticks)
        pyplot.setp(ax.get_xticklabels()[1::2], visible=False)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_yticks(y_ticks)
        pyplot.setp(ax.get_yticklabels()[1::2], visible=False)
        ax.set_ylim(y_range[0], y_range[1])

    figure.supxlabel(x_label)
    figure.supylabel(y_label)

    pyplot.savefig(output_path)
    pyplot.close(figure)


@click.command()
@click.option(
    "-d/-l",
    "--dark_background/--light_background",
    default=True,
    help="Use the pyplot `dark_background` style.",
)
@click.option(
    "-e",
    "--extension",
    type=click.STRING,
    default="pdf",
    show_default=True,
    help="File extension for output plots.",
)
@click.option(
    "-f",
    "--figure_width",
    type=click.FLOAT,
    default=4.25,
    show_default=True,
    help="Width of plots in inches.",
)
@click.option(
    "-h",
    "--figure_height",
    type=click.FLOAT,
    default=None,
    show_default=True,
    help="Height of plots in inches. Default is 0.75 times figure_width.",
)
@click.option(
    "-i",
    "--input_dir",
    type=click.STRING,
    default="results",
    show_default=True,
    help="Directory path containing benchmark results.",
)
@click.option(
    "-o",
    "--output_dir",
    type=click.STRING,
    default="plots",
    show_default=True,
    help="Directory path to which plots should be written.",
)
@click.option(
    "-s",
    "--font_size",
    type=click.INT,
    default=None,
    show_default=True,
    help="Font size in pt. Default is matplotlib rcParams.",
)
def main(
    dark_background,
    extension,
    figure_width,
    figure_height,
    input_dir,
    output_dir,
    font_size,
):

    if dark_background:
        pyplot.style.use("dark_background")

    # Reorder seaborn colorblind palette to avoid similar orange and red hues
    seaborn.set_palette(
        seaborn.color_palette(
            [
                seaborn.color_palette("colorblind")[i]
                for i in [0, 1, 2, 4, 8, 9, 7, 5, 6, 3]
                #for i in [0, 4, 8, 9, 7, 5, 6, 3]
            ]
        )
    )

    if figure_height is None:
        figure_size = tuple(figure_width * x for x in (1, 0.75))
    else:
        figure_size = (figure_width, figure_height)

    if font_size is not None:
        pyplot.rcParams.update({"font.size": font_size})

    N_replicas = 3
    N_windows = 24
    replicas = [str(replica) for replica in numpy.arange(1, N_replicas + 1)]
    windows = [f"{i:02d}" for i in range(N_windows)]

    for output_prefix in [
        "null-0.0.3-pair-opc3",
    ]:
        if output_prefix == "null-0.0.3-pair-opc3":
            wm_prefix = output_prefix.split("-", maxsplit=2)[2]

            ff_labels = {
                "Null-0.0.3-Pair-{wm_prefix.upper()}": output_prefix,
            }

        target_labels = {
            "Butane": "butane",
        }

        native_contacts_df = pandas.DataFrame()

        for target_label, target in target_labels.items():
            for ff_label, force_field in ff_labels.items():
                for replica in replicas:
                    for window in windows:
                        observable_path = Path(
                            input_dir,
                            f"{target}-{force_field}",
                            f"replica-{replica}",
                            f"window-{window}",
                            f"{target}-{force_field}-production-fraction-"
                            "native-contacts.dat",
                        )

                        window_df = pandas.read_csv(
                            observable_path,
                            sep="\s+",
                            skiprows=1,
                            names=["Step", "C-C-C-C dihedral (deg)"],
                        )
                        window_df["C-C-C-C dihedral (deg)"] *= numpy.rad2deg(1)
                        window_df["Replica"] = replica
                        window_df["Window"] = window
                        native_contacts_df = pandas.concat(
                            [native_contacts_df, window_df]
                        )

            native_contacts_df.to_csv(
                Path(
                    output_dir, f"{target}-{output_prefix}-dihedral-cv.dat"
                ),
            )

            _plot_histogram(
                plot_data_df=native_contacts_df,
                row_labels=replicas,
                output_path=Path(
                    output_dir,
                    f"{target}-{output_prefix}-dihedral-cv.{extension}",
                ),
                histogram_df_column="C-C-C-C dihedral (deg)",
                figure_size=figure_size,
                bin_width=1.5,
                category_labels=windows,
                category_df_column="Window",
                x_label="C-C-C-C dihedral (deg)",
                y_ticks=numpy.arange(0, 0.25, 0.04),
                y_range=(0, 0.24),
            )


#        if output_prefix == "ff14sb-opc3":
#            _plot_histogram(
#                plot_data_df=native_contacts_df,
#                row_labels=["1"],
#                output_path=Path(
#                    output_dir,
#                    f"windows-vs-histograms.{extension}",
#                ),
#                histogram_df_column="Fraction native contacts",
#                figure_size=figure_size,
#                bin_width=0.002,
#                category_labels=windows,
#                category_df_column="Window",
#                x_label="Fraction native contacts",
#                y_ticks=numpy.arange(0, 0.25, 0.04),
#                y_range=(0, 0.24),
#                shade_bins=True,
#            )

if __name__ == "__main__":
    main()
