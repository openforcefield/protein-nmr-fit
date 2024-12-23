import itertools
from pathlib import Path
from typing import TypedDict

import click
import loos
import numpy
import openmm
import pandas
from loos.pyloos import Trajectory
from openmm import app, unit
from proteinbenchmark import benchmark_targets, read_xml

helix_resids = {
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
}
hairpin_resids = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
}
loop_resids = {"10", "11", "21", "38", "39", "40", "41", "48", "49", "56"}


class NativeContactDict(TypedDict):
    chain_i: str
    resid_i: int
    name_i: str
    chain_j: str
    resid_j: int
    name_j: str
    native_distance: float


def get_native_contacts(
    native_topology_path: str,
    selection_string: str = "!hydrogen",
    sequence_cutoff: int = 3,
    distance_cutoff: float = 4.5,
):
    # Set up native topology
    topology = loos.createSystem(native_topology_path)
    # Select atoms that can contribute to native contacts
    contact_atoms = loos.selectAtoms(topology, selection_string)

    native_contacts = list()

    # Loop over pairs of atoms that can contribute to native contacts
    for atom_i, atom_j in itertools.combinations(contact_atoms, 2):
        # Skip pairs close in primary sequence
        if numpy.abs(atom_i.resid() - atom_j.resid()) <= sequence_cutoff:
            continue

        # Skip pairs far in space
        native_distance = atom_i.coords().distance(atom_j.coords())
        if native_distance > distance_cutoff:
            continue

        # Record chain, resid, and name of both atoms and their distance
        native_contacts.append(
            {
                "chain_i": atom_i.chainId(),
                "resid_i": atom_i.resid(),
                "name_i": atom_i.name(),
                "chain_j": atom_j.chainId(),
                "resid_j": atom_j.resid(),
                "name_j": atom_j.name(),
                "native_distance": native_distance,
            }
        )

    return native_contacts


def compute_fraction_native_contacts(
    native_contacts: list[NativeContactDict],
    topology_path: str,
    trajectory_path: str,
    smoothing_parameter: float = 5.0,
    contact_width: float = 1.8,
):
    topology = loos.createSystem(topology_path)

    # Get list of atom indices for native contacts
    native_contact_indices = list()
    for contact in native_contacts:
        selection_string = (
            f'chainid == "{contact["chain_i"]}" '
            f'&& resid == {contact["resid_i"]} && name == "{contact["name_i"]}"'
        )
        atom_i = loos.selectAtoms(topology, selection_string)
        if len(atom_i) != 1:
            raise ValueError(
                f"Selection string '{selection_string}' matches multiple atoms."
            )

        selection_string = (
            f'chainid == "{contact["chain_j"]}" '
            f'&& resid == {contact["resid_j"]} && name == "{contact["name_j"]}"'
        )
        atom_j = loos.selectAtoms(topology, selection_string)
        if len(atom_j) != 1:
            raise ValueError(
                f"Selection string '{selection_string}' matches multiple atoms."
            )

        native_contact_indices.append((atom_i[0].index(), atom_j[0].index()))

    native_contact_distances = contact_width * numpy.array(
        [contact["native_distance"] for contact in native_contacts]
    )
    contact_distances = numpy.zeros(native_contact_distances.size)

    trajectory = loos.pyloos.Trajectory(trajectory_path, topology)
    fraction_native_contacts = list()

    for frame in trajectory:
        for contact_index, atom_indices in enumerate(native_contact_indices):
            atom_i, atom_j = frame[atom_indices[0]], frame[atom_indices[1]]
            contact_distances[contact_index] = atom_i.coords().distance(atom_j.coords())
        contact_differences = contact_distances - native_contact_distances
        fraction_native_contacts.append(
            numpy.reciprocal(
                1 + numpy.exp(smoothing_parameter * contact_differences)
            ).mean()
        )

    return fraction_native_contacts


@click.command()
@click.option(
    "-f",
    "--force-field",
    type=click.STRING,
    default="ff14sb-tip3p",
    show_default=True,
    help="Name of force field used to sample the trajectory.",
)
@click.option(
    "-i",
    "--input_dir",
    type=click.STRING,
    default=Path("..", "results"),
    show_default=True,
    help="Directory path containing benchmark results.",
)
@click.option(
    "-o",
    "--output_dir",
    type=click.STRING,
    default="fraction-native-contacts",
    show_default=True,
    help="Directory path to write fraction of native contacts.",
)
@click.option(
    "-r",
    "--replica",
    type=click.STRING,
    default="1",
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
def main(
    force_field,
    input_dir,
    output_dir,
    replica,
    target,
):
    native_topology_path = str(benchmark_targets[target]["initial_pdb"])

    native_contacts = get_native_contacts(native_topology_path)

    with open(Path(output_dir, f"{target}-native-contacts.dat"), "w") as out_file:
        out_file.write(
            "# Chain_i Resid_i Name_i Chain_j Resid_j Name_j Native_distance"
        )
        for contact in native_contacts:
            out_file.write(
                f'\n{contact["chain_i"]:1s} {contact["resid_i"]:5d} '
                f'{contact["name_i"]:4s} {contact["chain_j"]:1s} '
                f'{contact["resid_j"]:5d} {contact["name_j"]:4s} '
                f'{contact["native_distance"]:12.8f}'
            )

    topology_path = str(
        Path(
            input_dir,
            f"{target}-{force_field}",
            "analysis",
            f"{target}-{force_field}-{replica}-reimaged.pdb",
        )
    )

    trajectory_path = str(
        Path(
            input_dir,
            f"{target}-{force_field}",
            "analysis",
            f"{target}-{force_field}-{replica}-reimaged.dcd",
        )
    )

    fraction_native_contacts = compute_fraction_native_contacts(
        native_contacts,
        topology_path,
        trajectory_path,
    )

    with open(
        Path(
            output_dir,
            f"{target}-{force_field}-{replica}-fraction-native-contacts.dat",
        ),
        "w",
    ) as out_file:
        out_file.write("# Frame Fraction_native_contacts")
        for frame_index, fraction in enumerate(fraction_native_contacts):
            out_file.write(f"\n{frame_index:6d} {fraction:10.8f}")


if __name__ == "__main__":
    main()
