from pathlib import Path

import click
import loos
import numpy
import openmm
import pandas
from loos.pyloos import Trajectory
from openmm import app
from openmm import unit as openmm_unit
from proteinbenchmark import (
    OpenMMSimulation,
    ProteinBenchmarkSystem,
    benchmark_targets,
    exists_and_not_empty,
    force_fields,
    read_xml,
    write_pdb,
)
from proteinbenchmark.simulation_parameters import *


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
def main(
    force_field,
    output_directory,
    replica,
    target,
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

    # Create a directory for this replica if it doesn't already exist
    replica_dir = Path(benchmark_system.base_path, f"replica-{replica:d}")
    replica_dir.mkdir(parents=True, exist_ok=True)

    replica_prefix = Path(replica_dir, benchmark_system.system_name)
    equil_prefix = f"{replica_prefix}-equilibration"

    # Saved state from the end of the equilibration simulation
    equilibrated_state = f"{equil_prefix}-1.xml"

    # Equilibrate at constant pressure and temperature
    if not exists_and_not_empty(equilibrated_state):
        print(f"Running NPT equilibration for system {benchmark_system.system_name}")

        # Get parameters for equilibration simulation
        if "equil_timestep" in target_parameters:
            equil_timestep = target_parameters["equil_timestep"]
        else:
            equil_timestep = EQUIL_TIMESTEP

        if "equil_traj_length" in target_parameters:
            equil_traj_length = target_parameters["equil_traj_length"]
        else:
            equil_traj_length = EQUIL_TRAJ_LENGTH

        if "equil_frame_length" in target_parameters:
            equil_frame_length = target_parameters["equil_frame_length"]
        else:
            equil_frame_length = EQUIL_FRAME_LENGTH

        if "equil_langevin_friction" in target_parameters:
            equil_langevin_friction = target_parameters["equil_langevin_friction"]
        else:
            equil_langevin_friction = EQUIL_LANGEVIN_FRICTION

        if "equil_barostat_frequency" in target_parameters:
            equil_barostat_frequency = target_parameters["equil_barostat_frequency"]
        else:
            equil_barostat_frequency = EQUIL_OPENMM_BAROSTAT_FREQUENCY

        # Initialize the equilibration simulation
        equilibration_dcd = f"{equil_prefix}.dcd"
        equilibration_state_data = f"{equil_prefix}.out"
        equilibration_checkpoint = f"{equil_prefix}.chk"

        equilibration_simulation = OpenMMSimulation(
            openmm_system_file=benchmark_system.parametrized_system,
            initial_pdb_file=benchmark_system.minimized_coords,
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
        )

        # Run equilibration
        equilibration_simulation.start_from_pdb()

    # 31 windows sampling Q = 1.0 to 0.4 in steps of 0.02
    umbrella_centers = numpy.linspace(1.0, 0.4, 31)

    # Parameters for constant velocity steered MD
    if "langevin_friction" in target_parameters:
        langevin_friction = target_parameters["langevin_friction"]
    else:
        langevin_friction = LANGEVIN_FRICTION

    if "barostat_frequency" in target_parameters:
        barostat_frequency = target_parameters["barostat_frequency"]
    else:
        barostat_frequency = OPENMM_BAROSTAT_FREQUENCY

    traj_length = 1.2 * openmm_unit.nanosecond
    timestep = 1.0 * openmm_unit.femtosecond
    steer_increment = 10
    frame_length = steer_increment * timestep
    steer_speed = (umbrella_centers[0] - umbrella_centers[-1]) / traj_length * 1.2

    # Set up constant velocity steered MD
    umbrella_system_path = str(
        Path(
            benchmark_system.setup_dir,
            f"{target}-{force_field}-umbrella-openmm-system.xml",
        )
    )
    umbrella_system = read_xml(umbrella_system_path)
    for force in umbrella_system.getForces():
        if isinstance(force, openmm.CustomCVForce):
            umbrella_force = force
    initial_pdb = app.PDBFile(benchmark_system.minimized_coords)

    # Set up BAOAB Langevin integrator from openmmtools with VRORV splitting
    integrator = openmm.LangevinMiddleIntegrator(
        temperature,
        langevin_friction.to_openmm(),
        timestep,
    )

    # Set up Monte Carlo barostat
    if pressure.value_in_unit(openmm_unit.atmosphere) > 0:
        umbrella_system.addForce(
            openmm.MonteCarloBarostat(
                pressure,
                temperature,
                barostat_frequency,
            )
        )

    # Create simulation
    simulation = app.Simulation(
        initial_pdb.topology,
        umbrella_system,
        integrator,
        openmm.Platform.getPlatformByName("CUDA"),
        {"Precision": "mixed"},
    )
    simulation.loadState(equilibrated_state)

    # Run constant velocity steered MD
    steering_center = umbrella_centers[0]
    window_index = 0
    for i in range(int(numpy.round(traj_length / frame_length))):
        simulation.step(steer_increment)
        fraction_native_contacts = umbrella_force.getCollectiveVariableValues(
            simulation.context
        )[0]

        # Decrement the center of the steering force
        steering_center -= steer_speed * frame_length
        simulation.context.setParameter("Q0", steering_center)

        # Save coordinates if the simulation has reached a window center
        if (
            window_index < len(umbrella_centers)
            and fraction_native_contacts <= umbrella_centers[window_index]
        ):
            print(
                f"{window_index:2d} {umbrella_centers[window_index]:4.2f} "
                f"{fraction_native_contacts:6.4f} {steering_center:6.4f}"
            )
            write_pdb(
                f"{replica_prefix}-window-{window_index:02d}.pdb",
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
            )
            window_index += 1


if __name__ == "__main__":
    main()
