from pathlib import Path

import click
import loos
from loos.pyloos import Trajectory
import numpy
from openff.units import unit
import pandas
from proteinbenchmark import (
    align_trajectory,
    exists_and_not_empty,
    force_fields,
    list_of_dicts_to_csv,
)


def align_downsampled_trajectory(
    topology_path: str,
    trajectory_path: str,
    output_prefix: str,
    output_selection: str = 'chainid == "A"',
    align_selection: str = None,
    reference_path: str = None,
    reference_selection: str = None,
):
    """
    Center and align a subset of atoms in a trajectory.

    Parameters
    ---------
    topology_path
        The path to the system topology, e.g. a PDB file.
    trajectory_path
        The path to the trajectory.
    output_prefix
        The prefix for the path to write the aligned topology and trajectory.
    output_selection
        LOOS selection string for atoms to write to the output. Default is all
        atoms.
    align_selection
        LOOS selection string for atoms to use for alignment. Default is output
        selection.
    reference_path
        The path to the structure used as a reference for alignment. Default is
        the system topology.
    reference_selection
        LOOS selection string for atoms to use for alignment in the reference
        structure. Default is align selection.
    """

    # Load topology and trajectory
    topology = loos.createSystem(topology_path)
    trajectory = Trajectory(trajectory_path, topology)

    # Set up reference structure for alignment
    if reference_path is None:
        reference = topology.copy()
    else:
        reference = loos.createSystem(reference_path)

    # Atom selections
    output_atoms = loos.selectAtoms(topology, output_selection)

    if align_selection is None:
        align_atoms = output_atoms
    else:
        align_atoms = loos.selectAtoms(topology, align_selection)

    if reference_selection is None:
        if align_selection is None:
            reference_atoms = loos.selectAtoms(reference, output_selection)
        else:
            reference_atoms = loos.selectAtoms(reference, align_selection)

    else:
        reference_atoms = loos.selectAtoms(reference, reference_selection)

    # Set up writer for output trajectory
    output_trajectory = loos.DCDWriter(f"{output_prefix}.dcd")

    first_frame = True

    for frame_index, frame in enumerate(trajectory):
        if frame_index % 10 != 9:
            continue

        # Align frame onto reference
        transform_matrix = align_atoms.superposition(reference_atoms)

        # Apply transformation to output atoms
        output_atoms.applyTransform(loos.loos.XForm(transform_matrix))

        # Recenter output atoms
        output_atoms.centerAtOrigin()

        # Write current frame for output atoms
        output_trajectory.writeFrame(output_atoms)

        # Write a PDB file of the first frame for use as a topology file
        if first_frame:
            first_frame = False
            pdb = loos.PDB.fromAtomicGroup(output_atoms)
            pdb.clearBonds()

            with open(f"{output_prefix}.pdb", "w") as pdb_file:
                pdb_file.write(str(pdb))


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
    default="butane",
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
    analysis_directory = Path(
        output_directory,
        f"{target}-{force_field}",
        "analysis",
    )
    analysis_directory.mkdir(parents=True, exist_ok=True)

    analysis_prefix = Path(
        analysis_directory,
        f"{target}-{force_field}-{replica}-{window_index:02d}",
    )

    reimaged_topology_path = f"{analysis_prefix}-reimaged.pdb"
    reimaged_trajectory_path = f"{analysis_prefix}-reimaged.dcd"
    frame_length = 100.0 * unit.picosecond

    # Align production trajectory
    if not exists_and_not_empty(reimaged_topology_path):
        print(
            f"Aligning production trajectory for system {target}-{force_field} "
            f"{replica} {window_index}"
        )

        topology_path = str(
            Path(
                output_directory,
                f"{target}-{force_field}",
                "setup",
                f"{target}-{force_field}-minimized.pdb",
            )
        )

        trajectory_path = str(
            Path(
                output_directory,
                f"{target}-{force_field}",
                f"replica-{replica:d}",
                f"window-{window_index:02d}",
                f"{target}-{force_field}-production.dcd",
            )
        )

        #align_downsampled_trajectory(
        align_trajectory(
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            output_prefix=f"{analysis_prefix}-reimaged",
            output_selection='chainid == "A"',
            align_selection='name =~ "^C(1|2|3)"',
            reference_path=topology_path,
        )

    # Measure dihedrals
    dihedrals_path = f"{analysis_prefix}-dihedrals.dat"

    if not exists_and_not_empty(dihedrals_path):
        print(
            f"Measuring dihedrals for system {target}-{force_field} {replica} "
            f"{window_index}"
        )

        # Load topology
        topology = loos.createSystem(reimaged_topology_path)

        # Select atoms for dihedrals
        atoms = list()

        for atom_name in ["C1x", "C2x", "C3x", "C4x"]:
            atom_selection = loos.selectAtoms(topology, f'name == "{atom_name}"')

            if len(atom_selection) == 0:
                raise ValueError(
                    f"Unable to select atom {atom_name} with resid "
                    f"{atom_resid} for dihedral {dihedral}."
                )

            atoms.append(atom_selection[0])

        # Set up trajectory
        trajectory = Trajectory(reimaged_trajectory_path, topology)
        frame_time = 0.0 * unit.picosecond
        fragment_index = 0
        dihedrals = list()

        # Load one frame into memory at a time
        for frame in trajectory:
            frame_time += frame_length

            frame_index = trajectory.index()
            frame_time_ns = frame_time.m_as(unit.nanosecond)

            # Measure dihedrals
            dihedral = loos.torsion(atoms[0], atoms[1], atoms[2], atoms[3])

            dihedrals.append(
                {
                    "Frame": frame_index,
                    "Time (ns)": frame_time_ns,
                    "Dihedral Name": "C-C-C-C",
                    "Dihedral (deg)": dihedral,
                }
            )

        list_of_dicts_to_csv(dihedrals, dihedrals_path)

    # Scalar couplings
    scalar_couplings_path = f"{analysis_prefix}-gaussian-observable.dat"
    time_series_path = f"{analysis_prefix}-gaussian-observable-time-series.dat"

    if not exists_and_not_empty(scalar_couplings_path):
        print(
            f"Computing Gaussian observable for system {target}-{force_field} "
            f"{replica} {window_index}"
        )

        # Make fake data for experimental observable that favors eclipsed
        # conformation at phi = 0
        # O(phi) = exp(-(phi - mu)^2 / sigma^2)
        experimental_observable = 1

        # Read time series of dihedrals
        dihedral_df = pandas.read_csv(dihedrals_path, index_col=0)

        # Compute observables
        observable_timeseries = list()
        computed_observables = list()

        # Compute cos(theta + delta)
        dihedral_angle = dihedral_df["Dihedral (deg)"].values

        # Compute estimate for Gaussian observable
        # <O> = <exp(-(phi - mu)^2 / sigma^2)>
        computed_observable = numpy.exp(-numpy.square(dihedral_angle / 15))

        # Get experimental uncertainty
        experiment_uncertainty = 1

        # Truncate experimental coupling to Karplus extrema
        truncated_experimental_observable = experimental_observable

        # Write time series of observable
        observable_timeseries.append(
            {
                "Frame": dihedral_df["Frame"],
                "Time (ns)": dihedral_df["Time (ns)"],
                "Observable": "3j_c_c",
                "Experiment": experimental_observable,
                "Experiment Uncertainty": experiment_uncertainty,
                "Truncated Experiment": truncated_experimental_observable,
                "Computed": computed_observable,
            }
        )

        # Write computed means of observable
        computed_observables.append(
            {
                "Observable": "3j_c_c",
                "Experiment": experimental_observable,
                "Experiment Uncertainty": experiment_uncertainty,
                "Truncated Experiment": truncated_experimental_observable,
                "Correlated Mean": computed_observable.mean(),
                "Correlated SEM": (
                    computed_observable.std(ddof=1) / numpy.sqrt(computed_observable.size)
                ),
            }
        )

        observable_timeseries_df = pandas.concat(
            [pandas.DataFrame(df) for df in observable_timeseries]
        ).reset_index(drop=True)
        observable_timeseries_df.to_csv(time_series_path)

        scalar_coupling_df = pandas.DataFrame(computed_observables)
        scalar_coupling_df.to_csv(scalar_couplings_path)

if __name__ == "__main__":
    main()
