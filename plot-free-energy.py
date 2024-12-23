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
    min_x = numpy.floor(plot_data_df[x_df_column].min() / 0.05) * 0.05
    max_x = numpy.ceil(plot_data_df[x_df_column].max() / 0.05) * 0.05
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
        x_ticks = numpy.round(numpy.arange(min_x, max_x + 0.025, 0.05), 1)

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
    pyplot.setp(ax.get_xticklabels()[0::2], visible=False)
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
    min_x = numpy.floor(plot_data_df[x_df_column].min() / 0.05) * 0.05
    max_x = numpy.ceil(plot_data_df[x_df_column].max() / 0.05) * 0.05
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
        x_ticks = numpy.round(numpy.arange(min_x + 0.05, max_x + 0.05, 0.1), 1)

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
            pyplot.setp(ax.get_xticklabels()[1::2], visible=False)
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
            elif "NMR-0.8" in row_label or "NMR-0.7" in row_label:
                a = row_label.split("-")
                ax.set_ylabel(
                    "-".join(a[:1]) + "\n" + "-".join(a[1:3]) + "\n"
                    + "-".join(a[3:-1]) + "\n" + "-".join(a[-1:])
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

    #    figure.legend(loc="outside upper center", ncol=2)

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
    replicas = [str(replica) for replica in numpy.arange(1, N_replicas + 1)]

    for output_prefix in [
        #"gb3-opc3",
        #"gb3-nmr-opc3",
        #"gb3-nmr-pred-opc3",
        #"gb3-nmr-0.8-opc3",
        #"gb3-nmr-0.8-pred-opc3",
        "gb3-nmr-0.8-cum-opc3",
    ]:
        if output_prefix == "gb3-opc3":
            ff_labels = {
                "ff14SB-OPC3": "ff14sb-opc3",
                "ff14SBonlysc-OPC3": "ff14sbonlysc-opc3",
                "Null-0.0.3-Pair-OPC3": "null-0.0.3-pair-opc3",
                "Specific-0.0.3-Pair-OPC3": "specific-0.0.3-pair-opc3",
                "Specific-0.0.3-SPair-OPC3": "specific-0.0.3-sage-pair-opc3",
            }

            target_labels = {
                "GB3": "gb3",
            }

        elif output_prefix == "gb3-nmr-opc3":
            ff_labels = {
                "ff14SB-OPC3": "ff14sb-opc3",
                "Null-QM-OPC3": "null-0.0.3-pair-opc3",
                "Null-NMR-0.8-1E4-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-opc3",
                "Null-NMR-0.8-1E4-2-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-2-opc3",
                "Null-NMR-0.8-1E4-3-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
                "Null-NMR-0.8-1E4-4-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-4-opc3",
                #"Null-NMR-1E5-OPC3": "null-0.0.3-pair-nmr-1e5-opc3",
                #"Null-NMR-1E5-2-OPC3": "null-0.0.3-pair-nmr-1e5-2-opc3",
                #"Null-NMR-1E5-3-OPC3": "null-0.0.3-pair-nmr-1e5-3-opc3",
                #"Null-NMR-1E4-OPC3": "null-0.0.3-pair-nmr-1e4-opc3",
                #"Null-NMR-1E3-OPC3": "null-0.0.3-pair-nmr-1e3-opc3",
                #"Null-NMR-1E2-OPC3": "null-0.0.3-pair-nmr-1e2-opc3",
                #"Null-Gen-1E5-OPC3": "null-0.0.3-pair-general-nmr-1e5-opc3",
            }

            target_labels = {
                "GB3": "gb3",
            }

        elif output_prefix == "gb3-nmr-pred-opc3":
            ff_labels = {
                "ff14SB-OPC3": "ff14sb-opc3",
                "Null-QM-OPC3": "null-0.0.3-pair-opc3",
                "Null-NMR-1E5-OPC3-Obs": "null-0.0.3-pair-nmr-1e5-opc3",
                "Null-NMR-1E5-2-OPC3-Obs": "null-0.0.3-pair-nmr-1e5-2-opc3",
                "Null-NMR-1E5-OPC3-Pred": [
                    "null-0.0.3-pair-opc3",
                    "null-0.0.3-pair-nmr-1e5-opc3",
                ],
                "Null-NMR-1E5-2-OPC3-Pred": [
                    "null-0.0.3-pair-nmr-1e5-opc3",
                    "null-0.0.3-pair-nmr-1e5-2-opc3",
                ],
                #"Null-NMR-1E5-3-OPC3": "null-0.0.3-pair-nmr-1e5-3-opc3",
            }

            target_labels = {
                "GB3": "gb3",
            }

        elif output_prefix == "gb3-nmr-0.8-opc3":
            ff_labels = {
                "ff14SB-OPC3": "ff14sb-opc3",
                "Null-QM-OPC3": "null-0.0.3-pair-opc3",
                "Null-NMR-0.8-1E4-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-opc3",
                "Null-NMR-0.8-1E4-2-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-2-opc3",
                "Null-NMR-0.8-1E4-3-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
                "Null-NMR-0.8-1E4-4-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-4-opc3",
            }

            target_labels = {
                "GB3": "gb3",
            }

        elif output_prefix == "gb3-nmr-0.8-pred-opc3":
            ff_labels = {
                "ff14SB-OPC3": "ff14sb-opc3",
                "Null-QM-OPC3": "null-0.0.3-pair-opc3",
                "Null-NMR-0.8-1E4-OPC3-Obs": "null-0.0.3-pair-nmr-0.8-1e4-opc3",
                "Null-NMR-0.8-1E4-2-OPC3-Obs": "null-0.0.3-pair-nmr-0.8-1e4-2-opc3",
                "Null-NMR-0.8-1E4-3-OPC3-Obs": "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
                "Null-NMR-0.8-1E4-4-OPC3-Obs": "null-0.0.3-pair-nmr-0.8-1e4-4-opc3",
                "Null-NMR-0.8-1E4-OPC3-Pred": [
                    "null-0.0.3-pair-opc3",
                    "null-0.0.3-pair-nmr-0.8-1e4-opc3",
                ],
                "Null-NMR-0.8-1E4-2-OPC3-Pred": [
                    "null-0.0.3-pair-nmr-0.8-1e4-opc3",
                    "null-0.0.3-pair-nmr-0.8-1e4-2-opc3",
                ],
                "Null-NMR-0.8-1E4-3-OPC3-Pred": [
                    "null-0.0.3-pair-nmr-0.8-1e4-2-opc3",
                    "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
                ],
                "Null-NMR-0.8-1E4-4-OPC3-Pred": [
                    "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
                    "null-0.0.3-pair-nmr-0.8-1e4-4-opc3",
                ],
            }

            target_labels = {
                "GB3": "gb3",
            }

        elif output_prefix == "gb3-nmr-0.8-cum-opc3":
            ff_labels = {
                "ff14SB-OPC3": "ff14sb-opc3",
                "Null-QM-OPC3": "null-0.0.3-pair-opc3",
                "Null-NMR-0.8-1E4-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-opc3",
                "Null-NMR-0.8-1E4-2-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-2-opc3",
                "Null-NMR-0.8-1E4-3-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
                "Null-NMR-0.8-1E4-3-Cu-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
            }

            target_labels = {
                "GB3": "gb3",
            }

        free_energy_df = pandas.DataFrame()

        for target_label, target in target_labels.items():
            for ff_label, force_field in ff_labels.items():
                #wham_free_energy_path = Path(
                #    input_dir,
                #    f"{target}-{force_field}",
                #    "analysis",
                #    f"{target}-{force_field}-wham-free-energy.dat",
                #)

                #wham_free_energy = numpy.loadtxt(wham_free_energy_path)

                #pooled_df = pandas.DataFrame(
                #    {
                #        "Fraction Native Contacts": wham_free_energy[:, 0],
                #        "Free Energy (kcal mol^-1)": wham_free_energy[:, 1],
                #        "Free Energy Uncertainty (kcal mol^-1)": wham_free_energy[:, 2],
                #    }
                #)

                #pooled_df["Force Field"] = f"WHAM {ff_label}"
                #pooled_df["Replica"] = "Pooled"
                #free_energy_df = pandas.concat([free_energy_df, pooled_df])

                if output_prefix.startswith("gb3-nmr-0.8"):
                    mbar_str = "mbar-0.8"
                elif output_prefix.startswith("gb3-nmr-0.7"):
                    mbar_str = "mbar-0.7"
                else:
                    mbar_str = "mbar"

                if ff_label.endswith("Pred"):
                    mbar_free_energy_path = Path(
                        input_dir,
                        f"{target}-{force_field[0]}",
                        "analysis",
                        f"{target}-{force_field[0]}-{mbar_str}-{force_field[1]}-"
                            "free-energy.dat",
                    )
                elif "Cu" in ff_label:
                    mbar_free_energy_path = Path(
                        input_dir,
                        f"{target}-{force_field}",
                        "analysis",
                        f"{target}-{force_field}-{mbar_str}-cum-free-energy.dat",
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
                        "Fraction Native Contacts",
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
                            "Fraction Native Contacts",
                            "Free Energy (kcal mol^-1)",
                            "Free Energy Uncertainty (kcal mol^-1)",
                        ],
                    )

                    replica_df["Force Field"] = ff_label
                    replica_df["Replica"] = replica
                    free_energy_df = pandas.concat([free_energy_df, replica_df])

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
                x_df_column="Fraction Native Contacts",
                y_df_column="Free Energy (kcal mol^-1)",
                y_uncertainty_df_column="Free Energy Uncertainty (kcal mol^-1)",
                figure_size=figure_size,
                column_df_column="Replica",
                x_label="Fraction native contacts",
                y_range = (0, 8),
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
                x_df_column="Fraction Native Contacts",
                y_df_column="Free Energy (kcal mol^-1)",
                y_uncertainty_df_column="Free Energy Uncertainty (kcal mol^-1)",
                figure_size=figure_size,
                x_label="Fraction native contacts",
                y_range = (0, 6),
            )


if __name__ == "__main__":
    main()
