import json
from pathlib import Path

import click
import numpy
import pandas
import seaborn
from matplotlib import pyplot


def _plot_pooled_free_energy(
    plot_data_df,
    category_labels: list[str],
    output_path: str,
    x_df_column: str,
    y_df_column: str,
    y_uncertainty_df_column: str,
    figure_size: tuple[float, float],
    category_df_column="Force Field",
    x_label: str = "Collective variable",
    y_label: str = "Free energy (kcal mol$^{-1}$)",
    x_ticks: list[float] = None,
    y_ticks: list[float] = None,
    x_range: tuple[float, float] = None,
    y_range: tuple[float, float] = None,
):
    #min_x = numpy.floor(plot_data_df[x_df_column].min() / 15) * 15
    min_x = -180.0
    max_x = numpy.ceil(plot_data_df[x_df_column].max() / 15) * 15
    max_y = (
        numpy.ceil(
            plot_data_df.loc[
                plot_data_df[y_df_column] != numpy.inf,
                y_df_column,
            ].max()
            / 0.5
        )
        * 0.5
    )

    if x_ticks is None:
        x_ticks = numpy.arange(min_x, max_x + 7.5, 30)

    if x_range is None:
        x_range = (min_x, max_x)

    if y_ticks is None:
        y_ticks = numpy.arange(0.0, max_y + 0.5, 1.0)

    if y_range is None:
        y_range = (0, max_y)

    figure = pyplot.figure(figsize=figure_size)
    ax = pyplot.gca()

    # Count number of predicted rather than observed category labels
    pred_color_offset = len(
        [label for label in category_labels if label.endswith("Pred")]
    )

    for category_index, category_label in enumerate(category_labels):
        subplot_df = plot_data_df[plot_data_df[category_df_column] == category_label]

        if subplot_df.shape[0] == 0:
            continue

        x_data = subplot_df[x_df_column].values
        y_data = subplot_df[y_df_column].values
        y_uncertainty = subplot_df[y_uncertainty_df_column].values

        if x_data.size == 0 or y_data.size == 0:
            continue

        # Set color index and dashed linestyle for predicted free energy surfaces
        if category_label.endswith("Pred"):
            color_index = category_index - pred_color_offset
            linestyle = "--"
        else:
            color_index = category_index
            linestyle = "-"

        category_color = seaborn.color_palette()[color_index % 10]

        ax.plot(
            x_data,
            y_data,
            label=category_label.replace("-OPC3", ""),
            linestyle=linestyle,
            color=category_color,
        )

        ax.fill_between(
            x_data,
            y_data - y_uncertainty,
            y_data + y_uncertainty,
            linewidth=0,
            linestyle=linestyle,
            color=category_color,
            alpha=0.5,
        )

    ax.set_xticks(x_ticks)
    pyplot.setp(ax.get_xticklabels()[1::2], visible=False)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_yticks(y_ticks)
    pyplot.setp(ax.get_yticklabels()[1::2], visible=False)
    ax.set_ylim(y_range[0], y_range[1])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    figure.legend(loc="outside upper center", ncol=2)

    pyplot.savefig(output_path)
    pyplot.close(figure)


def _plot_free_energy(
    plot_data_df,
    row_labels: list[str],
    column_labels: list[str],
    output_path: str,
    x_df_column: str,
    y_df_column: str,
    y_uncertainty_df_column: str,
    figure_size: tuple[float, float],
    row_df_column="Force Field",
    column_df_column="Replica",
    x_label: str = "Collective variable",
    y_label: str = "Free energy (kcal mol$^{-1}$)",
    x_ticks: list[float] = None,
    y_ticks: list[float] = None,
    x_range: tuple[float, float] = None,
    y_range: tuple[float, float] = None,
):
    #min_x = numpy.floor(plot_data_df[x_df_column].min() / 15) * 15
    min_x = -180.0
    max_x = numpy.ceil(plot_data_df[x_df_column].max() / 15) * 15
    max_y = (
        numpy.ceil(
            plot_data_df.loc[
                plot_data_df[y_df_column] != numpy.inf,
                y_df_column,
            ].max()
            / 0.5
        )
        * 0.5
    )

    if x_ticks is None:
        x_ticks = numpy.arange(min_x, max_x + 30, 60)

    if x_range is None:
        x_range = (min_x, max_x)

    if y_ticks is None:
        y_ticks = numpy.arange(0.0, max_y + 1.0, 2.0)

    if y_range is None:
        y_range = (0, max_y)

    figure, axes = pyplot.subplots(
        len(row_labels),
        len(column_labels),
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_index, row_label in enumerate(row_labels):
        row_color = seaborn.color_palette()[row_index % 10]

        for column_index, column_label in enumerate(column_labels):
            ax = axes[row_index, column_index]

            subplot_df = plot_data_df[
                (plot_data_df[row_df_column] == row_label)
                & (plot_data_df[column_df_column] == column_label)
            ]

            if subplot_df.shape[0] == 0:
                continue

            x_data = subplot_df[x_df_column].values
            y_data = subplot_df[y_df_column].values
            y_uncertainty = subplot_df[y_uncertainty_df_column].values

            if x_data.size == 0 or y_data.size == 0:
                continue

            ax.plot(
                x_data,
                y_data,
                color=row_color,
            )

            ax.fill_between(
                x_data,
                y_data - y_uncertainty,
                y_data + y_uncertainty,
                linewidth=0,
                color=row_color,
                alpha=0.5,
            )

            ax.set_xticks(x_ticks)
            pyplot.setp(
                [
                    ax.get_xticklabels()[i]
                    for i in range(len(ax.get_xticklabels()))
                    if i % 3 != 0
                ],
                visible=False,
            )
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_yticks(y_ticks)
            pyplot.setp(ax.get_yticklabels()[1::2], visible=False)
            ax.set_ylim(y_range[0], y_range[1])

            if "ff14SB" in row_label:
                ax.set_ylabel(row_label.replace("-", "\n", 1))
            elif "TIP3P-FB" in row_label:
                a = row_label.split("-")
                ax.set_ylabel(
                    "-".join(a[:1]) + "\n" + "-".join(a[1:-2]) + "\n" + "-".join(a[-2:])
                )
            else:
                a = row_label.split("-")
                ax.set_ylabel(
                    "-".join(a[:1]) + "\n" + "-".join(a[1:-1]) + "\n" + "-".join(a[-1:])
                )

    for ax in axes.flat:
        ax.label_outer()

    figure.supxlabel(x_label)
    figure.supylabel(y_label)

    #figure.legend(loc="outside upper center", ncol=2)

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
        "butane-opc3",
    ]:
        if output_prefix == "butane-opc3":
            ff_labels = {
                "Null-QM-OPC3": "null-0.0.3-pair-opc3",
                "Null-NMR-1E-3-OPC3-Obs": "null-0.0.3-pair-nmr-1e-3-opc3",
                #"Null-NMR-1E-3-2-OPC3-Obs": "null-0.0.3-pair-nmr-1e-3-cum-2-opc3",
                "Null-NMR-1E-3-Cu-2-OPC3-Obs": "null-0.0.3-pair-nmr-1e-3-cum-2-opc3",
                #"Null-NMR-1E-3-OPC3-Pred": [
                #    "null-0.0.3-pair-opc3",
                #    "null-0.0.3-pair-nmr-1e-3-opc3",
                #],
                #"Null-NMR-1E-3-Cu-2-OPC3-Pred": [
                #    "null-0.0.3-pair-nmr-1e-3-opc3",
                #    "null-0.0.3-pair-nmr-1e-3-cum-2-opc3",
                #],
            }

            target_labels = {
                "Butane": "butane",
            }

        free_energy_df = pandas.DataFrame()

        for target_label, target in target_labels.items():
            for ff_label, force_field in ff_labels.items():
                mbar_str = "mbar"

                if ff_label.endswith("Pred"):
                    if "Cu" in ff_label:
                        mbar_str = f"{mbar_str}-cum"
                    mbar_free_energy_path = Path(
                        input_dir,
                        f"{target}-{force_field[0]}",
                        "analysis",
                        f"{target}-{force_field[0]}-{mbar_str}-"
                            f"{force_field[1]}-free-energy.dat",
                    )
                else:
                    mbar_free_energy_path = Path(
                        input_dir,
                        f"{target}-{force_field}",
                        "analysis",
                        f"{target}-{force_field}-{mbar_str}-free-energy.dat",
                    )

                mbar_df = pandas.read_csv(
                    mbar_free_energy_path,
                    index_col=0,
                    header=0,
                    names=[
                        "C-C-C-C dihedral (deg)",
                        "Free Energy (kcal mol^-1)",
                        "Free Energy Uncertainty (kcal mol^-1)",
                    ],
                )

                mbar_df["Force Field"] = ff_label
                mbar_df["Replica"] = "Pooled"
                free_energy_df = pandas.concat([free_energy_df, mbar_df])

                for replica in replicas:
                    if ff_label.endswith("Pred"):
                        replica_free_energy_path = Path(
                            input_dir,
                            f"{target}-{force_field[0]}",
                            "analysis",
                            f"{target}-{force_field[0]}-{mbar_str}-{replica}-"
                                f"{force_field[1]}-free-energy.dat",
                        )
                    else:
                        replica_free_energy_path = Path(
                            input_dir,
                            f"{target}-{force_field}",
                            "analysis",
                            f"{target}-{force_field}-{mbar_str}-{replica}-free-"
                                "energy.dat",
                        )

                    if not replica_free_energy_path.exists():
                        continue

                    replica_df = pandas.read_csv(
                        replica_free_energy_path,
                        index_col=0,
                        header=0,
                        names=[
                            "C-C-C-C dihedral (deg)",
                            "Free Energy (kcal mol^-1)",
                            "Free Energy Uncertainty (kcal mol^-1)",
                        ],
                    )

                    replica_df["Force Field"] = ff_label
                    replica_df["Replica"] = replica
                    free_energy_df = pandas.concat([free_energy_df, replica_df])

            free_energy_df["C-C-C-C dihedral (deg)"] *= numpy.rad2deg(1)
            free_energy_df.to_csv(
                Path(output_dir, f"{output_prefix}-free-energy.dat"),
            )

            plot_ff_labels = free_energy_df[
                free_energy_df["Replica"] != "Pooled"
            ]["Force Field"].unique()

            _plot_free_energy(
                plot_data_df=free_energy_df,
                row_labels=plot_ff_labels,
                column_labels=replicas,
                output_path=Path(
                    output_dir,
                    f"{output_prefix}-free-energy.{extension}",
                ),
                x_df_column="C-C-C-C dihedral (deg)",
                y_df_column="Free Energy (kcal mol^-1)",
                y_uncertainty_df_column="Free Energy Uncertainty (kcal mol^-1)",
                figure_size=figure_size,
                column_df_column="Replica",
                x_label="C-C-C-C dihedral (deg)",
            )

            plot_ff_labels = free_energy_df[
                free_energy_df["Replica"] == "Pooled"
            ]["Force Field"].unique()

            _plot_pooled_free_energy(
                plot_data_df=free_energy_df[free_energy_df["Replica"] == "Pooled"],
                category_labels=plot_ff_labels,
                output_path=Path(
                    output_dir,
                    f"{output_prefix}-pooled-free-energy.{extension}",
                ),
                x_df_column="C-C-C-C dihedral (deg)",
                y_df_column="Free Energy (kcal mol^-1)",
                y_uncertainty_df_column="Free Energy Uncertainty (kcal mol^-1)",
                figure_size=figure_size,
                x_label="C-C-C-C dihedral (deg)",
            )


if __name__ == "__main__":
    main()
