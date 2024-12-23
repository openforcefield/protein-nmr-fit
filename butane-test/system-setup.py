from pathlib import Path

import click
import loos
import numpy
from openff.toolkit import (
    ForceField,
    Molecule,
    Topology,
    ToolkitRegistry,
    RDKitToolkitWrapper,
)
from openff.toolkit.utils import toolkit_registry_manager
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
from openff.units import unit as openff_unit
import openmm
from openmm import app, unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from proteinbenchmark import (
    force_fields,
    minimize_openmm,
    write_pdb,
    write_xml,
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
    "-t",
    "--target",
    type=click.STRING,
    default="butane",
    show_default=True,
    help="Name of benchmark target.",
)
def main(
    force_field,
    output_directory,
    target,
):
    # Set up system parameters
    force_field_dict = force_fields["null-0.0.3-pair-opc3"]
    force_field_file = str(Path("gaussian-force-fields", f"{force_field}.offxml"))
    water_model = force_field_dict["water_model"]
    water_model_file = force_field_dict["water_model_file"]

    # Generate a conformer for butane
    butane = Molecule.from_smiles("CCCC")
    butane.generate_conformers()

    # Set up OpenMM ForceField with SMIRNOFFTemplateGenerator
    smirnoff_generator = SMIRNOFFTemplateGenerator(molecules=butane)
    modeller_force_field = app.ForceField("amber/tip3p_standard.xml")
    modeller_force_field.registerTemplateGenerator(smirnoff_generator.generator)

    # Solvate butane molecule in water
    modeller = app.Modeller(
        butane.to_topology().to_openmm(),
        butane.conformers[0].to_openmm(),
    )

    modeller.addSolvent(
        forcefield=modeller_force_field,
        model="tip3p",
        padding=1.4 * unit.nanometer,
        boxShape="dodecahedron",
        ionicStrength=0 * unit.molar,
        neutralize=False,
    )

    # Get the number of water molecules and their positions
    water_positions = {}
    _oxygen = app.element.oxygen
    for chain in modeller.topology.chains():
        for residue in chain.residues():
            if residue.name == "HOH":
                for atom in residue.atoms():
                    if atom.element == _oxygen:
                        water_positions[residue] = modeller.positions[atom.index]

    n_water = len(water_positions)

    print(f"Added {n_water} waters")

    # Write solvated system to PDB file
    setup_directory = Path(output_directory, f"{target}-{force_field}", "setup")
    setup_directory.mkdir(parents=True, exist_ok=True)
    solvated_pdb_path = str(Path(
        setup_directory,
        f"{target}-{force_field}-solvated.pdb",
    ))
    write_pdb(solvated_pdb_path, modeller.topology, modeller.positions)

    # Assign parameters to the solvated system
    openff_topology = Topology.from_openmm(
        modeller.topology,
        unique_molecules=[butane, Molecule.from_smiles("O")],
    )
    openff_force_field = ForceField(force_field_file, water_model_file)

    with toolkit_registry_manager(
        ToolkitRegistry([RDKitToolkitWrapper, NAGLToolkitWrapper])
    ):
        openmm_system = openff_force_field.create_openmm_system(openff_topology)

    # Repartition hydrogen mass to bonded heavy atom
    hydrogen_mass = 3.0 * unit.dalton
    _hydrogen = app.element.hydrogen
    for atom1, atom2 in modeller.topology.bonds():
        if atom1.element == _hydrogen:
            (atom1, atom2) = (atom2, atom1)
        if (
            atom2.element == _hydrogen
            and atom1.element not in {_hydrogen, None}
            and atom2.residue.name != "HOH"
        ):
            transfer_mass = (
                hydrogen_mass - openmm_system.getParticleMass(atom2.index)
            )
            heavy_mass = (
                openmm_system.getParticleMass(atom1.index) - transfer_mass
            )
            openmm_system.setParticleMass(atom2.index, hydrogen_mass)
            openmm_system.setParticleMass(atom1.index, heavy_mass)

    # Write OpenMM system to XML file
    parametrized_system_path = str(Path(
        setup_directory,
        f"{target}-{force_field}-openmm-system.xml",
    ))
    write_xml(parametrized_system_path, openmm_system)

    # Minimize energy of solvated system with Cartesian restraints on
    # non-hydrogen solute atoms
    minimized_coords_path = str(Path(
        setup_directory,
        f"{target}-{force_field}-minimized.pdb",
    ))
    minimize_openmm(
        parametrized_system=parametrized_system_path,
        solvated_pdb_file=solvated_pdb_path,
        minimized_coords_file=minimized_coords_path,
        restraint_energy_constant=(
            1.0 * openff_unit.kilocalorie_per_mole / openff_unit.angstrom**2
        ),
        force_tolerance=(
            10 * openff_unit.kilojoules_per_mole / openff_unit.nanometer
        ),
    )

    # Set up C-C-C-C dihedral angle as a collective variable
    _carbon = app.element.carbon
    carbon_atoms = [
        atom.index for atom in modeller.topology.atoms()
        if atom.element == _carbon
    ]
    print(carbon_atoms)

    dihedral_cv = openmm.CustomTorsionForce("theta")
    dihedral_cv.addTorsion(*carbon_atoms)

    # Set up umbrella restraint force
    umbrella_energy_constant = 50.0 * unit.kilocalorie_per_mole
    umbrella_force = openmm.CustomCVForce(
        f"k * min(dphi, {2 * numpy.pi} - dphi)^2; dphi = abs(phi - phi0)"
    )
    umbrella_force.addGlobalParameter("k", umbrella_energy_constant)
    umbrella_force.addGlobalParameter("phi0", numpy.pi)
    umbrella_force.addCollectiveVariable("phi", dihedral_cv)

    # Load OpenMM system, add umbrella force, and save a copy
    umbrella_system_path = str(Path(
        setup_directory,
        f"{target}-{force_field}-umbrella-openmm-system.xml",
    ))

    openmm_system.addForce(umbrella_force)
    write_xml(umbrella_system_path, openmm_system)


if __name__ == "__main__":
    main()
