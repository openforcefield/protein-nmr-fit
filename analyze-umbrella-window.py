from pathlib import Path

import click
import numpy
from openff.units import unit
from proteinbenchmark import (
    ProteinBenchmarkSystem,
    align_trajectory,
    benchmark_targets,
    compute_scalar_couplings,
    exists_and_not_empty,
    force_fields,
    measure_dihedrals,
)


@click.command()
@click.option(
    "-f",
    "--force-field",
    type=click.STRING,
    default="null-0.0.3-pair-opc3",
    show_default=True,
    help="Name of force field used to sample the trajectory.",
)
@click.option(
    "-o",
    "--output_directory",
    type=click.STRING,
    default="results",
    show_default=True,
    help="Directory path to write umbrella sampling output.",
)
@click.option(
    "-r",
    "--replica",
    type=click.INT,
    default=1,
    show_default=True,
    help="Replica number for this target and force field to read.",
)
@click.option(
    "-t",
    "--target",
    type=click.STRING,
    default="gb3",
    show_default=True,
    help="Name of benchmark target.",
)
@click.option(
    "-w",
    "--window_index",
    type=click.INT,
    default=0,
    show_default=True,
    help="Zero-based index for umbrella window.",
)
def main(
    force_field,
    output_directory,
    replica,
    target,
    window_index,
):
    # Set up system parameters
    force_field_dict = force_fields[force_field]
    force_field_file = force_field_dict["force_field_file"]
    water_model = force_field_dict["water_model"]
    water_model_file = force_field_dict["water_model_file"]
    target_parameters = benchmark_targets[target]
    temperature = target_parameters["temperature"].to_openmm()
    pressure = target_parameters["pressure"].to_openmm()

    benchmark_system = ProteinBenchmarkSystem(
        output_directory,
        target,
        target_parameters,
        force_field,
        water_model,
        force_field_file,
        water_model_file=water_model_file,
    )

    analysis_dir = Path(benchmark_system.base_path, "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_prefix = Path(
        analysis_dir,
        f"{benchmark_system.system_name}-{replica}-{window_index:02d}",
    )

    reimaged_topology = f"{analysis_prefix}-reimaged.pdb"
    reimaged_trajectory = f"{analysis_prefix}-reimaged.dcd"
    frame_length = 100.0 * unit.picosecond

    # Align production trajectory
    if not exists_and_not_empty(reimaged_topology):
        print(
            "Aligning production trajectory for system "
            f"{benchmark_system.system_name} {replica} {window_index}"
        )

        trajectory_path = str(
            Path(
                benchmark_system.base_path,
                f"replica-{replica:d}",
                f"window-{window_index:02d}",
                f"{benchmark_system.system_name}-production.dcd",
            )
        )

        align_trajectory(
            topology_path=benchmark_system.minimized_coords,
            trajectory_path=trajectory_path,
            output_prefix=f"{analysis_prefix}-reimaged",
            output_selection='chainid == "A"',
            align_selection='name == "CA"',
            reference_path=benchmark_system.initial_pdb,
        )

    # Measure dihedrals
    dihedrals = f"{analysis_prefix}-dihedrals.dat"

    if not exists_and_not_empty(dihedrals):
        print(
            f"Measuring dihedrals for system {benchmark_system.system_name} "
            f"{replica} {window_index}"
        )

        fragment_index = measure_dihedrals(
            topology_path=reimaged_topology,
            trajectory_path=reimaged_trajectory,
            frame_length=frame_length,
            output_path=dihedrals,
        )

        if fragment_index > 0:
            merge_csvs(dihedrals)

    # Scalar couplings
    scalar_couplings = f"{analysis_prefix}-scalar-couplings.dat"
    time_series_output_path = f"{analysis_prefix}-scalar-couplings-time-series.dat"

    if not exists_and_not_empty(scalar_couplings):
        print(
            f"Computing scalar couplings for system "
            f"{benchmark_system.system_name} {replica} {window_index}"
        )

        experimental_observables = target_parameters["observables"]["scalar_couplings"][
            "observable_path"
        ]

        compute_scalar_couplings(
            observable_path=experimental_observables,
            dihedrals_path=dihedrals,
            output_path=scalar_couplings,
            time_series_output_path=time_series_output_path,
        )


if __name__ == "__main__":
    main()
