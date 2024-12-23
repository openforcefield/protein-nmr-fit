from pathlib import Path

import click
import numpy
import openmm
from openmm import app
from openmm import unit as openmm_unit
from proteinbenchmark import (
    OpenMMSimulation,
    ProteinBenchmarkSystem,
    benchmark_targets,
    exists_and_not_empty,
    force_fields,
)
from proteinbenchmark.simulation_parameters import *


class OpenMMUmbrellaSimulation(OpenMMSimulation):
    """OpenMMSimulation with a harmonic CustomCVForce."""

    def __init__(
        self,
        openmm_system_file: str,
        initial_pdb_file: str,
        dcd_reporter_file: str,
        state_reporter_file: str,
        checkpoint_file: str,
        save_state_prefix: str,
        temperature: unit.Quantity,
        pressure: unit.Quantity,
        langevin_friction: unit.Quantity,
        barostat_frequency: int,
        timestep: unit.Quantity,
        traj_length: unit.Quantity,
        frame_length: unit.Quantity,
        checkpoint_length: unit.Quantity,
        save_state_length: unit.Quantity,
        umbrella_energy_constant: unit.Quantity,
        window_center: float,
        cv_reporter_file: str | None = None,
    ):
        """
        Initializes the simulation parameters and checks units.

        Parameters
        ----------
        openmm_system_file
            The path to the parametrized OpenMM system as a serialized XML.
        initial_pdb_file
            Path to PDB file used to set initial coordinates.
        dcd_reporter_file
            Path to DCD file to write trajectory coordinates.
        state_reporter_file
            Path to file to write state data, e.g. energies and temperature.
        checkpoint_file
            Path to file to write binary checkpoints.
        save_state_prefix
            Path prefix for files to write serialized simulation states.
        temperature
            The target temperature of the Langevin thermostat.
        pressure
            The target pressure of the Monte Carlo barostat.
        langevin_friction
            The collision frequency of the Langevin integrator.
        barostat_frequency
            The number of steps between attempted pressure changes for the Monte
            Carlo barostat.
        timestep
            The timestep for the Langevin integrator.
        traj_length
            The length of the trajectory (N_steps = traj_length / timestep).
        frame_length
            The amount of time between writing coordinates and state data to
            disk.
        checkpoint_length
            The amount of time between writing binary checkpoints to disk.
        save_state_length
            The amount of time between writing serialized simulation states to
            disk.
        umbrella_energy_constant
            The energy constant for the harmonic umbrella restraint.
        window_center
            The center of the umbrella restraint for this window.
        cv_reporter_file
            Path to file to write the collective variable.
        """

        super().__init__(
            openmm_system_file,
            initial_pdb_file,
            dcd_reporter_file,
            state_reporter_file,
            checkpoint_file,
            save_state_prefix,
            temperature,
            pressure,
            langevin_friction,
            barostat_frequency,
            timestep,
            traj_length,
            frame_length,
            checkpoint_length,
            save_state_length,
        )

        self.umbrella_energy_constant = umbrella_energy_constant
        self.window_center = window_center
        self.cv_reporter_file = cv_reporter_file
        print(
            "    umbrella_energy_constant "
            f"{umbrella_energy_constant.value_in_unit(openmm_unit.kilocalorie_per_mole):.2f}"
            f"\n    window_center {window_center:.2f}"
        )

    def setup_simulation(
        self,
        return_pdb: bool = False,
    ):
        """
        Set up an OpenMM simulation with a Langevin integrator and a Monte Carlo
        barostat.

        Parameters
        ----------
        return_pdb
            Return OpenMM PDBFile as well as OpenMM Simulation.
        """

        if return_pdb:
            simulation, initial_pdb = super().setup_simulation(return_pdb=True)
        else:
            simulation = super().setup_simulation()

        simulation.context.setParameter("k", self.umbrella_energy_constant)
        simulation.context.setParameter("phi0", self.window_center)

        if return_pdb:
            return simulation, initial_pdb
        else:
            return simulation

    def start_from_pdb(self, save_state_file: str):
        """
        Start a new simulation initializing positions from a PDB and velocities
        to random samples from a Boltzmann distribution at the simulation
        temperature. Set box vectors from a serialized simulation state.

        Parameters
        ----------
        save_state_file
            Path to the serialized simulation state.
        """

        # Create an OpenMM simulation
        simulation, initial_pdb = self.setup_simulation(return_pdb=True)

        # Initialize box bectors from the serialized state
        with open(save_state_file, "r") as xml_file:
            state = openmm.XmlSerializer.deserialize(xml_file.read())
        simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())

        # Initialize positions from the topology PDB
        simulation.context.setPositions(initial_pdb.positions)

        # Initialize velocities to random samples from a Boltzmann distribution
        simulation.context.setVelocitiesToTemperature(self.temperature)

        # Run dynamics
        self.run_dynamics(simulation, append=False)

    def run_dynamics(
        self,
        simulation: app.Simulation,
        append: bool = False,
    ):
        """
        Run dynamics for a simulation and write output.

        Parameters
        ----------
        simulation
            An OpenMM Simulation object.
        append
            Append to DCD and state data reporters instead of overwriting them.
        """

        # Set up reporters for DCD trajectory coordinates, state data, and
        # binary checkpoints
        dcd_reporter = app.DCDReporter(
            self.dcd_reporter_file, self.output_frequency, append=append
        )
        state_reporter = app.StateDataReporter(
            self.state_reporter_file,
            self.output_frequency,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            speed=True,
            separator=" ",
            append=append,
        )
        checkpoint_reporter = app.CheckpointReporter(
            self.checkpoint_file, self.checkpoint_frequency
        )

        simulation.reporters.extend([dcd_reporter, state_reporter, checkpoint_reporter])

        # Set up reporter for collective variable values
        if self.cv_reporter_file is not None:
            for force in simulation.system.getForces():
                if isinstance(force, openmm.CustomCVForce):
                    umbrella_force = force

            if not append:
                with open(self.cv_reporter_file, "w") as out_file:
                    out_file.write('#"Step" "CV"\n')

        # Get current index of serialized simulation state files
        save_state_index = 0
        if append:
            save_state_dir = Path(self.save_state_prefix).parent
            glob_prefix = Path(self.save_state_prefix).name

            for save_state_file in save_state_dir.glob(f"{glob_prefix}-*.xml"):
                file_index = int(save_state_file.stem.split("-")[-1])
                if file_index > save_state_index:
                    save_state_index = file_index

        # Run dynamics until the desired number of steps is reached
        while simulation.currentStep < self.n_steps:
            steps_remaining = self.n_steps - simulation.currentStep
            steps_to_take = min(self.output_frequency, steps_remaining)

            # Run dynamics
            simulation.step(steps_to_take)

            if self.cv_reporter_file is not None:
                # Write collective variable value
                cv_value = umbrella_force.getCollectiveVariableValues(
                    simulation.context
                )[0]
                with open(self.cv_reporter_file, "a") as out_file:
                    out_file.write(f"{simulation.currentStep:d} {cv_value:.8f}\n")

            if simulation.currentStep % self.save_state_frequency == 0:
                # Write serialized simulation state
                save_state_index += 1
                save_state_file = f"{self.save_state_prefix}-{save_state_index}.xml"
                simulation.saveState(save_state_file)


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
    umbrella_energy_constant = 50.0 * openmm_unit.kilocalorie_per_mole

    # 24 windows sampling phi = -180 deg to 165 deg in steps of 15 deg
    umbrella_centers = numpy.deg2rad(numpy.linspace(-180, 165, 24))
    window_center = umbrella_centers[window_index]

    # Set up system parameters
    temperature = 298.0 * openmm_unit.kelvin
    pressure = 1.0 * openmm_unit.atmosphere

    setup_directory = Path(
        output_directory,
        f"{target}-{force_field}",
        "setup",
    )
    replica_directory = Path(
        output_directory,
        f"{target}-{force_field}",
        f"replica-{replica:d}",
    )

    umbrella_system_path = str(Path(
        setup_directory,
        f"{target}-{force_field}-umbrella-openmm-system.xml",
    ))
    window_initial_pdb_path = str(Path(
        replica_directory,
        f"{target}-{force_field}-window-{window_index:02d}.pdb",
    ))
    steered_md_equilibrated_state = str(Path(
        replica_directory,
        f"{target}-{force_field}-equilibration-1.xml",
    ))

    # Create a directory for this window if it doesn't already exist
    window_directory = Path(
        replica_directory,
        f"window-{window_index:02d}",
    )
    window_directory.mkdir(parents=True, exist_ok=True)

    window_prefix = Path(window_directory, f"{target}-{force_field}")
    equil_prefix = f"{window_prefix}-equilibration"
    prod_prefix = f"{window_prefix}-production"

    # Saved state from the end of the equilibration simulation
    equilibrated_state = f"{equil_prefix}-1.xml"

    # Equilibrate at constant pressure and temperature
    if not exists_and_not_empty(equilibrated_state):
        print(f"Running NPT equilibration for system {target}-{force_field}")

        # Get parameters for equilibration simulation
        equil_timestep = EQUIL_TIMESTEP
        equil_traj_length = EQUIL_TRAJ_LENGTH
        equil_frame_length = EQUIL_FRAME_LENGTH
        equil_langevin_friction = EQUIL_LANGEVIN_FRICTION
        equil_barostat_frequency = EQUIL_OPENMM_BAROSTAT_FREQUENCY

        # Initialize the equilibration simulation
        equilibration_dcd = f"{equil_prefix}.dcd"
        equilibration_state_data = f"{equil_prefix}.out"
        equilibration_checkpoint = f"{equil_prefix}.chk"

        equilibration_simulation = OpenMMUmbrellaSimulation(
            openmm_system_file=umbrella_system_path,
            initial_pdb_file=window_initial_pdb_path,
            dcd_reporter_file=equilibration_dcd,
            state_reporter_file=equilibration_state_data,
            checkpoint_file=equilibration_checkpoint,
            save_state_prefix=equil_prefix,
            temperature=temperature,
            pressure=pressure,
            langevin_friction=equil_langevin_friction.to_openmm(),
            barostat_frequency=equil_barostat_frequency,
            timestep=equil_timestep.to_openmm(),
            traj_length=equil_traj_length.to_openmm(),
            frame_length=equil_frame_length.to_openmm(),
            checkpoint_length=equil_traj_length.to_openmm(),
            save_state_length=equil_traj_length.to_openmm(),
            umbrella_energy_constant=umbrella_energy_constant,
            window_center=window_center,
        )

        # Run equilibration
        equilibration_simulation.start_from_pdb(steered_md_equilibrated_state)

    # Parameters for production simulation
    langevin_friction = LANGEVIN_FRICTION
    barostat_frequency = OPENMM_BAROSTAT_FREQUENCY

    timestep = 4.0 * openmm_unit.femtosecond
    traj_length = 1000.0 * openmm_unit.nanosecond
    frame_length = 100.0 * openmm_unit.picosecond
    checkpoint_length = frame_length * 100
    save_state_length = checkpoint_length * 10

    production_dcd = f"{prod_prefix}.dcd"
    production_state_data = f"{prod_prefix}.out"
    production_checkpoint = f"{prod_prefix}.chk"
    production_cv_file = f"{prod_prefix}-fraction-native-contacts.dat"

    production_simulation = OpenMMUmbrellaSimulation(
        openmm_system_file=umbrella_system_path,
        initial_pdb_file=window_initial_pdb_path,
        dcd_reporter_file=production_dcd,
        state_reporter_file=production_state_data,
        checkpoint_file=production_checkpoint,
        save_state_prefix=prod_prefix,
        temperature=temperature,
        pressure=pressure,
        langevin_friction=langevin_friction.to_openmm(),
        barostat_frequency=barostat_frequency,
        timestep=timestep,
        traj_length=traj_length,
        frame_length=frame_length,
        checkpoint_length=checkpoint_length,
        save_state_length=save_state_length,
        umbrella_energy_constant=umbrella_energy_constant,
        window_center=window_center,
        cv_reporter_file=production_cv_file,
    )

    # Run production
    if not exists_and_not_empty(production_checkpoint):
        # Start production simulation, initializing positions and
        # velocities to the final state from the equilibration simulation
        production_simulation.start_from_save_state(equilibrated_state)

    else:
        # Resume from a previous production checkpoint
        production_simulation.resume_from_checkpoint()


if __name__ == "__main__":
    main()
