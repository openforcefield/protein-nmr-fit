import json
from pathlib import Path

import click
import numpy
import pandas
import seaborn
from matplotlib import pyplot


def _plot_target_fraction_native_contacts(
    fnc_df: pandas.DataFrame,
    output_path: str,
    figure_size: tuple[float, float],
    y_label: str,
    frame_step: float = 0.1,
    max_time: float = 10000.0,
    max_y: float = None,
):

    if max_time is None:
        max_time = 10000.0

    ff_labels = [
        label for label in fnc_df.columns if (label != "Step" and label != "Time (ns)")
    ]

    plot_data = fnc_df[(fnc_df["Time (ns)"] < max_time) & (fnc_df["Step"] % 100 == 0)]

    if max_y is None:
        max_y = numpy.amax([plot_data[ff_label].max() for ff_label in ff_labels])

    figure, axes = pyplot.subplots(3, 3, figsize=figure_size)

    row_index = -1

    for ff_label in ff_labels:

        replica = int(ff_label.split("-")[-1])

        if replica == 1:
            column_index = -1
            row_index += 1

        column_index += 1

        ax = axes[row_index, column_index]

        ax.plot(
            plot_data["Time (ns)"] / 1000,
            plot_data[ff_label],
            color=seaborn.color_palette()[row_index],
        )

        ax.set_xlim(0, (max_time + frame_step) / 1000)
        ax.set_xticks(numpy.arange(0, (max_time + frame_step) / 1000, 2))
        ax.set_ylim(0, max_y)
        ax.set_yticks(numpy.arange(0, max_y + 0.1, 0.2))
        if "ff14SB" in ff_label:
            ax.set_ylabel(ff_label[:-2].replace("-", "\n", 1))
        elif "TIP3P-FB" in ff_label:
            a = ff_label[:-2].split("-")
            ax.set_ylabel(
                "-".join(a[:1]) + "\n" + "-".join(a[1:-2]) + "\n" + "-".join(a[-2:])
            )
        else:
            a = ff_label[:-2].split("-")
            ax.set_ylabel(
                "-".join(a[:1]) + "\n" + "-".join(a[1:-1]) + "\n" + "-".join(a[-1:])
            )

        # Hide alternating y tick labels
        # pyplot.setp(ax.get_yticklabels()[1::2], visible=False)

    for ax in axes.flat:
        ax.label_outer()

    figure.supxlabel("Time ($\mu$s)")
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
    default="fraction-native-contacts",
    show_default=True,
    help="Directory path containing benchmark results.",
)
@click.option(
    "-o",
    "--output_dir",
    type=click.STRING,
    default="fraction-native-contacts",
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
@click.option(
    "-t",
    "--max_time",
    type=click.FLOAT,
    default=None,
    show_default=True,
    help="Maximum time at which to truncate RMSD plots.",
)
def main(
    dark_background,
    extension,
    figure_width,
    figure_height,
    input_dir,
    output_dir,
    font_size,
    max_time,
):

    if dark_background:
        pyplot.style.use("dark_background")

    # Reorder seaborn colorblind palette to avoid similar orange and red hues
    seaborn.set_palette(
        seaborn.color_palette(
            [
                seaborn.color_palette("colorblind")[i]
                for i in [0, 1, 2, 4, 8, 9, 7, 5, 6, 3]
                #                for i in [0, 4, 8, 9, 7, 5, 6, 3]
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
    replicas = numpy.arange(1, N_replicas + 1)

    for output_prefix in [
        "0.0.3-tip3p",
        "0.0.3-opc3",
        "0.0.3-tip3p-fb",
        "0.0.3-opc",
        "0.0.3-ai-tip3p",
        "0.0.3-ai-opc3",
        "0.0.3-ai-tip3p-fb",
        "0.0.3-ai-opc",
        "0.0.3-nagl-tip3p",
        "0.0.3-nagl-opc3",
        "0.0.3-nagl-tip3p-fb",
        "0.0.3-nagl-opc",
    ]:
        if output_prefix.startswith("0.0.3-ai"):
            wm_prefix = output_prefix.split("-", maxsplit=2)[2]

            ff_labels = {
                f"ff14SB-{wm_prefix.upper()}": f"ff14sb-{wm_prefix}",
                f"Null-0.0.3-SP-{wm_prefix.upper()}": f"null-{output_prefix}",
                f"Specific-0.0.3-SP-{wm_prefix.upper()}": f"specific-{output_prefix}",
            }

        elif output_prefix.startswith("0.0.3-nagl"):
            wm_prefix = output_prefix.split("-", maxsplit=2)[2]

            ff_labels = {
                f"ff14SB-{wm_prefix.upper()}": f"ff14sb-{wm_prefix}",
                f"Null-0.0.3-NAGL-{wm_prefix.upper()}": f"null-0.0.3-nagl-{wm_prefix}",
                f"Null-0.0.3-Pair-{wm_prefix.upper()}": f"null-0.0.3-pair-{wm_prefix}",
            }

        elif output_prefix.startswith("0.0.3"):
            wm_prefix = output_prefix.split("-", maxsplit=1)[1]

            ff_labels = {
                f"ff14SB-{wm_prefix.upper()}": f"ff14sb-{wm_prefix}",
                f"Null-{output_prefix.upper()}": f"null-{output_prefix}",
                f"Specific-{output_prefix.upper()}": f"specific-{output_prefix}",
            }

        target_labels = {
            "GB3": "gb3",
        }

        for target_label, target in target_labels.items():
            first_ff = True

            for ff_label, force_field in ff_labels.items():
                first_replica = True

                for replica in replicas:
                    observable_path = Path(
                        input_dir,
                        f"{target}-{force_field}-{replica}-fraction-native-contacts.dat",
                    )

                    observable_df = pandas.read_csv(
                        observable_path,
                        sep="\s+",
                        skiprows=1,
                        names=["Step", f"{ff_label}-{replica}"],
                    )

                    if first_replica:
                        replica_df = observable_df
                        first_replica = False

                    else:
                        replica_df = pandas.merge(replica_df, observable_df, on="Step")

                if first_ff:
                    fnc_df = replica_df
                    first_ff = False

                else:
                    fnc_df = pandas.merge(fnc_df, replica_df, on="Step")

            # Time (ns) per frame in trajectory
            frame_step = 0.1
            fnc_df["Time (ns)"] = fnc_df["Step"] * frame_step

            fnc_df.to_csv(
                Path(
                    output_dir, f"{target}-{output_prefix}-fraction-native-contacts.dat"
                ),
            )

            _plot_target_fraction_native_contacts(
                fnc_df=fnc_df,
                output_path=Path(
                    output_dir,
                    f"{target}-{output_prefix}-fraction-native-contacts.{extension}",
                ),
                figure_size=figure_size,
                y_label="Fraction native contacts",
                frame_step=frame_step,
                max_time=max_time,
                max_y=1.0,
            )


if __name__ == "__main__":
    main()
