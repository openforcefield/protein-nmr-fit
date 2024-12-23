import itertools
from pathlib import Path
from typing import TypedDict

import click
import loos
import numpy
import openmm
from openmm import app, unit
from proteinbenchmark import (
    ProteinBenchmarkSystem,
    benchmark_targets,
    force_fields,
    read_xml,
    write_xml,
)


class NativeContactDict(TypedDict):
    index_i: int
    index_j: int
    native_distance: unit.Quantity


def get_native_contacts(
    native_topology_path: str,
    native_contacts_file: str,
    selection_string: str = "!hydrogen",
    sequence_cutoff: int = 3,
    distance_cutoff: unit.Quantity = 0.45 * unit.nanometer,
):
    distance_cutoff = distance_cutoff.value_in_unit(unit.nanometer)

    # Set up native topology
    pdb = app.PDBFile(native_topology_path)
    topology = pdb.topology
    positions = pdb.positions.value_in_unit(unit.nanometer)

    # Set up periodic distance function
    box_vectors = topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    periodic_distance = app.internal.compiled.periodicDistance(box_vectors)

    # Select atoms that can contribute to native contacts
    _hydrogen = app.element.hydrogen
    contact_atoms = [
        atom
        for atom in topology.atoms()
        if atom.residue.chain.id == "A" and atom.element != _hydrogen
    ]

    # Loop over pairs of atoms that can contribute to native contacts
    native_contacts = list()
    with open(native_contacts_file, "w") as out_file:
        out_file.write(
            "# Chain_i Resid_i Name_i Chain_j Resid_j Name_j Native_distance\n"
        )

        for atom_i, atom_j in itertools.combinations(contact_atoms, 2):
            # Skip pairs close in primary sequence
            sequence_distance = numpy.abs(
                int(atom_i.residue.id) - int(atom_j.residue.id)
            )
            if sequence_distance <= sequence_cutoff:
                continue

            # Skip pairs far in space
            native_distance = periodic_distance(
                positions[atom_i.index], positions[atom_j.index]
            )
            if native_distance > distance_cutoff:
                continue

            # Write chain, resid, and name of both atoms and their distance
            out_file.write(
                f"{atom_i.residue.chain.id:1s} {atom_i.residue.id:5s} "
                f"{atom_i.name:4s} {atom_j.residue.chain.id:1s} "
                f"{atom_j.residue.id:5s} {atom_j.name:4s} "
                f"{native_distance:12.8f}\n"
            )

            # Record OpenMM system indices of both atoms and their distance
            native_contacts.append(
                {
                    "index_i": atom_i.index,
                    "index_j": atom_j.index,
                    "native_distance": native_distance * unit.nanometer,
                }
            )

    return native_contacts


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
    "-t",
    "--target",
    type=click.STRING,
    default="gb3",
    show_default=True,
    help="Name of benchmark target.",
)
def main(
    force_field,
    output_directory,
    target,
):
    # Set up system parameters
    force_field_dict = force_fields[force_field]
    force_field_file = force_field_dict["force_field_file"]
    water_model = force_field_dict["water_model"]
    water_model_file = force_field_dict["water_model_file"]
    target_parameters = benchmark_targets[target]

    benchmark_system = ProteinBenchmarkSystem(
        output_directory,
        target,
        target_parameters,
        force_field,
        water_model,
        force_field_file,
        water_model_file=water_model_file,
    )

    # Build coordinates, solvate, parametrize, and minimize energy
    benchmark_system.setup()

    # Get native contacts from solvated PDB so that heavy atoms have the same
    # coords as the initial PDB but atom indices match the parametrized system
    native_topology_path = str(
        Path(
            benchmark_system.setup_dir,
            f"{target}-{force_field}-solvated.pdb",
        )
    )
    native_contacts_file = str(
        Path(
            benchmark_system.setup_dir,
            f"{target}-{force_field}-native-contacts.dat",
        )
    )
    native_contacts = get_native_contacts(native_topology_path, native_contacts_file)

    # Set up fraction of native contacts as a collective variable
    # Q = 1/N sum_i 1 / (1 + exp(a * (r_i - b * r_0,i)))
    # Q = 1/N sum_i (1 - tanh(a/2 * (r_i - b * r_0,i))) / 2
    smoothing_parameter = 50.0 / unit.nanometer
    contact_width = 1.8
    fraction_native_contacts = openmm.CustomBondForce("Z * (1 - tanh(a * (r - r0)))")
    fraction_native_contacts.addGlobalParameter("Z", 0.5 / len(native_contacts))
    fraction_native_contacts.addGlobalParameter("a", smoothing_parameter / 2)
    fraction_native_contacts.addPerBondParameter("r0")
    fraction_native_contacts.setUsesPeriodicBoundaryConditions(True)

    for contact in native_contacts:
        fraction_native_contacts.addBond(
            contact["index_i"],
            contact["index_j"],
            [contact_width * contact["native_distance"]],
        )

    # Set up umbrella restraint force
    umbrella_energy_constant = 5000.0 * unit.kilocalorie_per_mole
    umbrella_force = openmm.CustomCVForce("k * (Q - Q0)^2")
    umbrella_force.addGlobalParameter("k", umbrella_energy_constant)
    umbrella_force.addGlobalParameter("Q0", 1.0)
    umbrella_force.addCollectiveVariable("Q", fraction_native_contacts)

    # Load OpenMM system, add umbrella force, and save a copy
    umbrella_system_path = str(
        Path(
            benchmark_system.setup_dir,
            f"{target}-{force_field}-umbrella-openmm-system.xml",
        )
    )
    openmm_system = read_xml(benchmark_system.parametrized_system)
    openmm_system.addForce(umbrella_force)
    write_xml(umbrella_system_path, openmm_system)


if __name__ == "__main__":
    main()
