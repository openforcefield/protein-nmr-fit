import json
from pathlib import Path

import click
import numpy
from openmm import unit
import pandas
from proteinbenchmark import benchmark_targets
import seaborn
from matplotlib import pyplot


def _plot_folding_free_energy(
    folding_cv: dict[numpy.typing.ArrayLike],
    weights: dict[numpy.typing.ArrayLike],
    RT: float,
    output_path: str,
    figure_size: tuple[float, float],
    x_label: str = "Folded state cutoff",
    y_label: str = "Folding free energy (kcal mol$^{-1}$)",
    x_ticks: list[float] = None,
    y_ticks: list[float] = None,
    x_range: tuple[float, float] = None,
    y_range: tuple[float, float] = None,
):

    min_x = max([x.min() for x in folding_cv.values()])
    max_x = min([x.max() for x in folding_cv.values()])
    min_x = (numpy.floor(min_x / 0.05) + 1) * 0.05
    max_x = (numpy.ceil(max_x / 0.05) - 1) * 0.05

    cutoff_values = numpy.arange(min_x, max_x + 0.001, 0.002)

    min_y, max_y = 0.0, 0.0

    figure = pyplot.figure(figsize=figure_size)
    ax = pyplot.gca()

    for category_index, category_label in enumerate(folding_cv.keys()):
        folding_free_energy = numpy.zeros(cutoff_values.size)

        for cutoff_index, cutoff_value in enumerate(cutoff_values):
            folded_samples = folding_cv[category_label] > cutoff_value
            folded_probability = weights[category_label][folded_samples].sum()
            unfolded_probability = weights[category_label][~folded_samples].sum()
            folding_free_energy[cutoff_index] = (
                -RT * numpy.log(folded_probability / unfolded_probability)
            )

        min_y = min(min_y, folding_free_energy.min())
        max_y = max(max_y, folding_free_energy.max())

        category_color = seaborn.color_palette()[category_index % 10]

        ax.plot(
            cutoff_values,
            folding_free_energy,
            label=category_label.replace("-OPC3", ""),
            linestyle="-",
            color=category_color,
        )

    min_y = numpy.floor(min_y / 2) * 2
    max_y = numpy.ceil(max_y / 2) * 2

    if x_ticks is None:
        x_ticks = numpy.round(numpy.arange(min_x, max_x + 0.025, 0.05), 1)

    if x_range is None:
        x_range = (min_x, max_x)

    if y_ticks is None:
        y_ticks = numpy.arange(min_y, max_y + 0.5, 1.0)

    if y_range is None:
        y_range = (min_y, max_y)

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
        "gb3-nmr-opc3",
        #"gb3-nmr-0.8-opc3",
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
                "Null-NMR-0.8-1E4-4-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
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

        elif output_prefix == "gb3-nmr-0.8-opc3":
            ff_labels = {
                "ff14SB-OPC3": "ff14sb-opc3",
                "Null-QM-OPC3": "null-0.0.3-pair-opc3",
                "Null-NMR-0.8-1E4-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-opc3",
                "Null-NMR-0.8-1E4-2-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-2-opc3",
                "Null-NMR-0.8-1E4-3-OPC3": "null-0.0.3-pair-nmr-0.8-1e4-3-opc3",
            }

            target_labels = {
                "GB3": "gb3",
            }

        fraction_native_contacts = dict()
        mbar_weights = dict()

        for target_label, target in target_labels.items():
            temperature = benchmark_targets[target]["temperature"]
            RT = unit.MOLAR_GAS_CONSTANT_R * temperature.to_openmm()
            RT = RT.value_in_unit(unit.kilocalorie_per_mole)

            for ff_label, force_field in ff_labels.items():
                if output_prefix.startswith("gb3-nmr-0.8"):
                    mbar_str = "mbar-0.8"
                else:
                    mbar_str = "mbar"

                mbar_samples_path = Path(
                    input_dir,
                    f"{target}-{force_field}",
                    "analysis",
                    f"{target}-{force_field}-{mbar_str}-samples.dat",
                )

                mbar_df = pandas.read_csv(
                    mbar_samples_path,
                    index_col=0,
                )

                fraction_native_contacts[ff_label] = mbar_df["Fraction Native Contacts"].values
                unnormalized_weights = 1.0 / mbar_df["MBAR Weight Denominator"].values
                mbar_weights[ff_label] = unnormalized_weights / unnormalized_weights.sum()

            _plot_folding_free_energy(
                folding_cv=fraction_native_contacts,
                weights=mbar_weights,
                RT=RT,
                output_path=Path(
                    output_dir,
                    f"{output_prefix}-folding-free-energy.{extension}",
                ),
                figure_size=figure_size,
                x_label="Folded state cutoff",
            )


if __name__ == "__main__":
    main()
