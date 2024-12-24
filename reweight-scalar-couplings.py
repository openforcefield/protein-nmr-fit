from pathlib import Path

import click
import loos
import numpy
import openmm
import pandas
from loos.pyloos import Trajectory
from openmm import unit
from proteinbenchmark.benchmark_targets import benchmark_targets
from proteinbenchmark.force_fields import force_fields
from proteinbenchmark.simulation_parameters import NONBONDED_CUTOFF, VDW_SWITCH_WIDTH
from proteinbenchmark.system_setup import assign_parameters
from proteinbenchmark.utilities import read_xml


class ReweightReference:
    """
    Score a force field by reweighting NMR observables sampled in a reference
    simulation.
    """

    def __init__(
        self,
        result_directory: str,
        force_field_name: str,
        observable_list: list[str],
        target_list: list[str],
        truncate_observables: bool,
    ):
        """
        Setup the time series of estimated observables and the paths to the
        OpenMM System, topology, and trajectory for the reference simulation.

        Parameters
        ----------
        result_directory
            Top-level directory containing simulation results.
        force_field_name
            Name of the force field used to sample the observable time series.
        observable_list
            List of observable names.
        target_list
            List of target names.
        truncate_observables
            Truncate experimental scalar couplings to extrema of the Karplus
            curve.
        """
        self.result_directory = result_directory
        self.force_field_name = force_field_name
        self.target_list = target_list

        self.get_target_betas()

        self.set_up_observables(
            observable_list,
            truncate_observables,
        )

        N_targets = len(target_list)
        self.reweighting_potential = [None for _ in range(N_targets)]
        self.uncorrelated_reweighting_potential = [None for _ in range(N_targets)]

    def __call__(
        self,
        query_system_paths: list[str],
        target_indices_to_skip: list[int] = None,
    ):
        """
        Compute the chi^2 value to quantify agreement between computed and
        experimental observables.

        Parameters
        ----------
        query_system_paths
            List of paths to OpenMM systems parametrized with query force field.
        target_indices_to_skip
            Indices of targets to skip for evaluation of the objective function.
        """

        return self.compute_chi_square_value(query_system_paths, target_indices_to_skip)

    def get_target_betas(self):
        """
        Read ensemble temperature for observable targets and compute
        thermodynamic beta, i.e. (k_B T)^-1.
        """

        target_betas = list()
        for target in self.target_list:
            target_temperature = benchmark_targets[target]["temperature"]
            RT = unit.MOLAR_GAS_CONSTANT_R * target_temperature.to_openmm()
            target_betas.append(1.0 / RT)

        self.target_betas = target_betas

    def set_up_observables(
        self,
        observable_list: list[str],
        truncate_observables: bool,
    ):
        """
        Read the uncorrelated sample indices and weight denominator from MBAR
        and the time series of estimated observables.

        Parameters
        ----------
        observable_list
            List of observable names.
        truncate_observables
            Truncate experimental scalar couplings to extrema of the Karplus
            curve.
        """

        # List of columns to read for observable DataFrames
        experiment_column = (
            "Truncated Experiment" if truncate_observables else "Experiment"
        )
        df_columns = [
            "Frame",
            "Observable",
            "Resid",
            "Resname",
            experiment_column,
            "Experiment Uncertainty",
            "Computed",
        ]

        # Sample indices and MBAR weight denominators by target
        self.mbar_samples = list()
        self.mbar_uncorrelated_samples = list()

        # Sample estimates of observables by target
        self.sampled_observables = list()
        self.uncorrelated_observables = list()

        # Experimental observables and uncertainties by target`
        self.experimental_observables = list()
        self.experimental_variances = list()

        print(
            "Reference_Force_Field             Target         Chi^2     "
            "(StDev)   N_eff  (StDev)"
        )

        for target in self.target_list:
            target_directory = Path(
                self.result_directory,
                f"{target}-{self.force_field_name}",
                "analysis",
            )

            # Read sample indices and MBAR weight denominators
            mbar_samples_path = Path(
                target_directory,
                f"{target}-{self.force_field_name}-mbar-0.8-samples.dat",
            )
            mbar_df = pandas.read_csv(
                mbar_samples_path,
                index_col=0,
                usecols=lambda column: column != "Fraction Native Contacts",
            )

            # Read uncorrelated sample indices and MBAR weight denominators
            mbar_uncorrelated_samples_path = Path(
                target_directory,
                f"{target}-{self.force_field_name}-mbar-0.8-uncorrelated-samples.dat",
            )
            mbar_uncorrelated_df = pandas.read_csv(
                mbar_uncorrelated_samples_path,
                index_col=0,
                usecols=lambda column: column != "Fraction Native Contacts",
            )

            # Read observables for uncorrelated sample indices in each window
            target_observable_df = pandas.DataFrame()
            target_uncorrelated_observable_df = pandas.DataFrame()

            for window in mbar_df["Window"].unique():
                window_mbar_df = mbar_df[mbar_df["Window"] == window]
                for replica in window_mbar_df["Replica"].unique():
                    window_uncorrelated_sample_indices = mbar_uncorrelated_df.loc[
                        (mbar_uncorrelated_df["Replica"] == replica)
                        & (mbar_uncorrelated_df["Window"] == window),
                        "Indices",
                    ].values

                    observable_path = Path(
                        target_directory,
                        f"{target}-{self.force_field_name}-{replica}-"
                        f"{window:02d}-scalar-couplings-time-series.dat",
                    )
                    observable_df = pandas.read_csv(
                        observable_path,
                        usecols=df_columns,
                    )

                    # All samples, correlated
                    observable_df = observable_df[
                        observable_df["Observable"].isin(observable_list)
                    ]
                    target_observable_df = pandas.concat(
                        [target_observable_df, observable_df]
                    )

                    # Uncorrelated samples
                    observable_df = observable_df[
                        observable_df["Frame"].isin(window_uncorrelated_sample_indices)
                    ]
                    target_uncorrelated_observable_df = pandas.concat(
                        [target_uncorrelated_observable_df, observable_df]
                    )

            # Create a (N_observables, N_samples) numpy array of computed values
            # of observables for each sample
            index_columns = [
                "Observable",
                "Resid",
                "Resname",
                "Truncated Experiment",
                "Experiment Uncertainty",
            ]
            target_observable_df.set_index(index_columns, inplace=True)
            target_observable_df.sort_index(inplace=True)
            target_uncorrelated_observable_df.set_index(index_columns, inplace=True)
            target_uncorrelated_observable_df.sort_index(inplace=True)
            observable_groups = target_observable_df.index.unique()

            N_observables = observable_groups.size
            N_samples = mbar_df.shape[0]
            N_uncorrelated_samples = mbar_uncorrelated_df.shape[0]

            target_experimental_observables = numpy.zeros(N_observables)
            target_experimental_uncertainties = numpy.zeros(N_observables)
            target_sampled_observables = numpy.zeros((N_observables, N_samples))
            target_uncorrelated_observables = numpy.zeros(
                (N_observables, N_uncorrelated_samples)
            )

            for observable_index, observable_group in enumerate(observable_groups):
                target_experimental_observables[observable_index] = observable_group[3]
                target_experimental_uncertainties[observable_index] = observable_group[
                    4
                ]
                target_sampled_observables[observable_index] = target_observable_df.loc[
                    observable_group, "Computed"
                ].values
                target_uncorrelated_observables[observable_index] = (
                    target_uncorrelated_observable_df.loc[
                        observable_group,
                        "Computed",
                    ].values
                )

            self.mbar_samples.append(mbar_df)
            self.mbar_uncorrelated_samples.append(mbar_uncorrelated_df)
            self.experimental_observables.append(target_experimental_observables)
            self.experimental_variances.append(
                numpy.square(target_experimental_uncertainties)
            )
            self.sampled_observables.append(target_sampled_observables)
            self.uncorrelated_observables.append(target_uncorrelated_observables)

            # Get chi^2 value and number of effective samples for unbiased state
            mbar_weights = 1.0 / mbar_df["MBAR Weight Denominator"].values
            mbar_weights /= mbar_weights.sum()

            estimated_observables = numpy.sum(
                mbar_weights * target_sampled_observables,
                axis=1,
            )
            reference_chi_square = numpy.mean(
                numpy.square(estimated_observables - target_experimental_observables)
                / numpy.square(target_experimental_uncertainties)
            )
            reference_effective_samples = int(
                numpy.round(1.0 / numpy.square(mbar_weights).sum())
            )

            # Get uncertainties for chi^2 and number of effective samples from
            # bootstrapping
            bootstrap_mbar_weights = (
                1.0
                / mbar_uncorrelated_df.loc[
                    :,
                    mbar_uncorrelated_df.columns.str.startswith(
                        "MBAR Weight Denominator"
                    ),
                ].values
            )
            bootstrap_mbar_weights /= bootstrap_mbar_weights.sum(axis=0)
            N_bootstraps = bootstrap_mbar_weights.shape[1]

            bootstrap_observables = numpy.zeros(
                (N_observables, N_uncorrelated_samples, N_bootstraps)
            )
            for bootstrap_index in range(N_bootstraps):
                bootstrap_observables[:, :, bootstrap_index] = (
                    target_uncorrelated_observables[
                        :,
                        mbar_uncorrelated_df[
                            f"Bootstrap Sample Indices {bootstrap_index}"
                        ].values,
                    ]
                )

            bootstrap_estimated_observables = numpy.sum(
                bootstrap_mbar_weights * bootstrap_observables,
                axis=1,
            )
            bootstrap_chi_square = numpy.mean(
                numpy.square(
                    bootstrap_estimated_observables.T - target_experimental_observables
                )
                / numpy.square(target_experimental_uncertainties),
                axis=1,
            )
            bootstrap_effective_samples = 1.0 / numpy.square(
                bootstrap_mbar_weights
            ).sum(axis=0)

            chi_square_uncertainty = bootstrap_chi_square.std(ddof=1)
            effective_samples_uncertainty = bootstrap_effective_samples.std(ddof=1)

            print(
                f"{self.force_field_name:33s} {target:14s} "
                f"{reference_chi_square:9.4f} ({chi_square_uncertainty:7.4}) "
                f"{reference_effective_samples:6d} "
                f"({effective_samples_uncertainty:5.1f})"
            )

    def compute_chi_square_value(
        self,
        query_system_paths: list[str],
        target_indices_to_skip: list[int],
    ):
        """
        Compute the chi^2 value to quantify agreement between computed and
        experimental observables.

        Parameters
        ----------
        query_system_paths
            List of paths to OpenMM systems parametrized with query force field.
        target_indices_to_skip
            Indices of targets to skip for evaluation of the objective function.
        """

        if len(query_system_paths) != len(self.target_list):
            raise ValueError(
                f"Different number of query systems ({len(query_system_paths)}) "
                f"and reference systems ({len(self.target_list)})"
            )

        chi_square = list()
        effective_samples = list()
        chi_square_uncertainty = list()
        effective_samples_uncertainty = list()

        for target_index, target in enumerate(self.target_list):
            if (
                target_indices_to_skip is not None
                and target_index in target_indices_to_skip
            ):
                continue

            query_system_path = query_system_paths[target_index]
            beta = self.target_betas[target_index]
            target_samples_df = self.mbar_samples[target_index]
            target_uncorrelated_samples_df = self.mbar_uncorrelated_samples[
                target_index
            ]
            target_experimental_observables = self.experimental_observables[
                target_index
            ]
            target_experimental_variances = self.experimental_variances[target_index]
            target_sampled_observables = self.sampled_observables[target_index]
            target_uncorrelated_observables = self.uncorrelated_observables[
                target_index
            ]

            # Set up target topology, OpenMM Systems, and OpenMM Contexts
            target_directory = Path(
                self.result_directory,
                f"{target}-{self.force_field_name}",
            )
            topology_path = Path(
                target_directory,
                "setup",
                f"{target}-{self.force_field_name}-minimized.pdb",
            )
            reference_system_path = Path(
                target_directory,
                "setup",
                f"{target}-{self.force_field_name}-openmm-system.xml",
            )

            topology = loos.createSystem(str(topology_path))
            reference_system = read_xml(str(reference_system_path))
            query_system = read_xml(query_system_path)
            reference_context = openmm.Context(
                reference_system,
                openmm.VerletIntegrator(1.0 * unit.femtosecond),
                openmm.Platform.getPlatformByName("CUDA"),
                {"Precision": "mixed"},
            )
            query_context = openmm.Context(
                query_system,
                openmm.VerletIntegrator(1.0 * unit.femtosecond),
                openmm.Platform.getPlatformByName("CUDA"),
                {"Precision": "mixed"},
            )
            loos_to_openmm = 1.0 * unit.angstrom / unit.nanometer

            # Loop over replicas and windows to get trajectories
            N_samples = target_samples_df.shape[0]
            reweighting_potential = numpy.zeros(N_samples)
            total_sample_index = 0

            N_uncorrelated_samples = target_uncorrelated_samples_df.shape[0]
            uncorrelated_reweighting_potential = numpy.zeros(N_uncorrelated_samples)
            total_uncorrelated_sample_index = 0

            for replica in target_samples_df["Replica"].unique():
                replica_samples_df = target_samples_df[
                    target_samples_df["Replica"] == replica
                ]
                replica_uncorrelated_samples_df = target_uncorrelated_samples_df[
                    target_uncorrelated_samples_df["Replica"] == replica
                ]

                for window in replica_samples_df["Window"].unique():
                    window_uncorrelated_sample_indices = (
                        replica_uncorrelated_samples_df.loc[
                            replica_uncorrelated_samples_df["Window"] == window,
                            "Indices",
                        ].values
                    )

                    trajectory_path = str(
                        Path(
                            target_directory,
                            f"replica-{replica}",
                            f"window-{window:02d}",
                            f"{target}-{self.force_field_name}-production.dcd",
                        )
                    )
                    trajectory = Trajectory(trajectory_path, topology)

                    # Loop over uncorrelated sample indices  and compute the
                    # reweighting potential
                    for frame in trajectory:
                        # Set periodic box vectors in OpenMM context from trajectory
                        d = frame.periodicBox()[0] * loos_to_openmm
                        box_vectors = numpy.array(
                            [
                                [d, 0, 0],
                                [0, d, 0],
                                [d / 2, d / 2, d / numpy.sqrt(2)],
                            ]
                        )
                        reference_context.setPeriodicBoxVectors(*box_vectors)
                        query_context.setPeriodicBoxVectors(*box_vectors)

                        coords = frame.getCoords() * loos_to_openmm
                        reference_context.setPositions(coords)
                        query_context.setPositions(coords)

                        reference_state = reference_context.getState(getEnergy=True)
                        reference_energy = reference_state.getPotentialEnergy()
                        query_state = query_context.getState(getEnergy=True)
                        query_energy = query_state.getPotentialEnergy()
                        energy_difference = beta * (query_energy - reference_energy)

                        reweighting_potential[total_sample_index] = energy_difference
                        total_sample_index += 1

                        if trajectory.index() in window_uncorrelated_sample_indices:
                            uncorrelated_reweighting_potential[
                                total_uncorrelated_sample_index
                            ] = energy_difference
                            total_uncorrelated_sample_index += 1

            # Offset the reweighting potential so that the lowest value is
            # zero and we don't get NaNs when we exponentiate it
            reweighting_potential_offset = reweighting_potential.min()
            reweighting_potential -= reweighting_potential_offset
            uncorrelated_reweighting_potential -= reweighting_potential_offset

            self.reweighting_potential[target_index] = reweighting_potential
            self.uncorrelated_reweighting_potential[target_index] = (
                uncorrelated_reweighting_potential
            )

            # Compute MBAR weights for reweighting from mixture distribution
            mbar_weights = (
                numpy.exp(-reweighting_potential)
                / target_samples_df["MBAR Weight Denominator"].values
            )

            # W(t) = exp(-U_rw(t)) / Z(t) / ( sum_t exp(-U_rw(t)) / Z(t) )
            # N_eff = (sum_t W(t))^2 / sum_t W(t)^2
            mbar_weight_normalization = mbar_weights.sum()
            if mbar_weight_normalization == 0.0:
                mbar_weights = numpy.zeros(N_samples)
                N_effective_samples = 0.0
            else:
                mbar_weights = mbar_weights / mbar_weight_normalization
                N_effective_samples = 1.0 / numpy.square(mbar_weights).sum()

            # < O_j > = sum_t W(t) * O_j(t)
            reweighted_estimates = numpy.sum(
                mbar_weights * target_sampled_observables,
                axis=1,
            )

            # chi^2 = sum_j (< O_j > - O_j,exp )^2 / sigma_j,exp^2
            target_chi_square = numpy.mean(
                numpy.square(reweighted_estimates - target_experimental_observables)
                / target_experimental_variances
            )

            # Get uncertainties from bootstrapping over uncorrelated samples
            N_bootstraps = len(
                [
                    column
                    for column in target_uncorrelated_samples_df
                    if column.startswith("Bootstrap Sample Indices")
                ]
            )
            bootstrap_boltzmann_factors = numpy.zeros(
                (N_uncorrelated_samples, N_bootstraps)
            )
            bootstrap_observables = numpy.zeros(
                (
                    len(target_experimental_observables),
                    N_uncorrelated_samples,
                    N_bootstraps,
                )
            )

            uncorrelated_boltzmann_factors = numpy.exp(
                -uncorrelated_reweighting_potential
            )
            for bootstrap_index in range(N_bootstraps):
                bootstrap_sample_indices = target_uncorrelated_samples_df[
                    f"Bootstrap Sample Indices {bootstrap_index}"
                ].values
                bootstrap_boltzmann_factors[:, bootstrap_index] = (
                    uncorrelated_boltzmann_factors[bootstrap_sample_indices]
                )
                bootstrap_observables[:, :, bootstrap_index] = (
                    target_uncorrelated_observables[:, bootstrap_sample_indices]
                )

            bootstrap_mbar_weights = (
                bootstrap_boltzmann_factors
                / target_uncorrelated_samples_df.loc[
                    :,
                    target_uncorrelated_samples_df.columns.str.startswith(
                        "MBAR Weight Denominator"
                    ),
                ].values
            )
            bootstrap_weight_normalization = bootstrap_mbar_weights.sum(axis=0)
            if numpy.any(bootstrap_weight_normalization == 0.0):
                bootstrap_mbar_weights = numpy.zeros(bootstrap_mbar_weights.shape)
                bootstrap_effective_samples = numpy.zeros(N_bootstraps)
            else:
                bootstrap_mbar_weights = (
                    bootstrap_mbar_weights / bootstrap_weight_normalization
                )
                bootstrap_effective_samples = 1.0 / numpy.square(
                    bootstrap_mbar_weights
                ).sum(axis=0)

            bootstrap_estimated_observables = numpy.sum(
                bootstrap_mbar_weights * bootstrap_observables,
                axis=1,
            )
            bootstrap_chi_square = numpy.mean(
                numpy.square(
                    bootstrap_estimated_observables.T - target_experimental_observables
                )
                / target_experimental_variances,
                axis=1,
            )

            target_chi_square_uncertainty = bootstrap_chi_square.std(ddof=1)
            target_effective_samples_uncertainty = bootstrap_effective_samples.std(
                ddof=1
            )

            chi_square.append(target_chi_square)
            effective_samples.append(N_effective_samples)
            chi_square_uncertainty.append(target_chi_square_uncertainty)
            effective_samples_uncertainty.append(target_effective_samples_uncertainty)

        return (
            chi_square,
            chi_square_uncertainty,
            effective_samples,
            effective_samples_uncertainty,
        )


@click.command()
@click.option(
    "-f",
    "--force-field-name",
    default="ff14sb-opc3",
    show_default=True,
    type=click.STRING,
    help="Name of force field for sampling force field.",
)
@click.option(
    "-i",
    "--input-directory",
    default="results",
    show_default=True,
    type=click.STRING,
    help="Directory path containing MBAR analysis, time series of observables,"
    "   and trajectories.",
)
@click.option(
    "-o",
    "--output-directory",
    default="reweight-scalar-couplings",
    show_default=True,
    type=click.STRING,
    help="Directory path to write query OpenMM systems.",
)
@click.option(
    "-q",
    "--query-force-fields",
    default="ff14sb-opc3,ff14sbonlysc-opc3",
    show_default=True,
    type=click.STRING,
    help="Comma-separated list of query force field names.",
)
@click.option(
    "-t/-r",
    "--truncate-observables/--retain-observables",
    default=True,
    help="Truncate scalar couplings to the extrema of the Karplus curve.",
)
def main(
    force_field_name,
    input_directory,
    output_directory,
    query_force_fields,
    truncate_observables,
):

    query_force_fields = query_force_fields.strip("'\"").split(",")

    target_list = ["gb3"]
    N_targets = len(target_list)

    observable_list = ["3j_hn_cb", "3j_hn_co", "3j_hn_ha"]

    # Set up reference simulation for reweighting
    reweight_reference = ReweightReference(
        input_directory,
        force_field_name,
        observable_list,
        target_list,
        truncate_observables,
    )

    # Get chi^2 value for query force fields
    print(
        "\nQuery_Force_Field                 Target         Chi^2     "
        "(StDev)   N_eff  (StDev)"
    )

    for query_ff in query_force_fields:
        if query_ff == force_field_name:
            query_system_paths = [
                Path(
                    input_directory,
                    f"{target}-{force_field_name}",
                    "setup",
                    f"{target}-{force_field_name}-openmm-system.xml",
                )
                for target in target_list
            ]

        else:
            query_system_paths = list()
            for target_index, target in enumerate(target_list):
                query_system_path = Path(
                    output_directory,
                    f"{target}-{query_ff}-openmm-system.xml",
                )
                query_system_paths.append(str(query_system_path))

                target_dir = Path(input_directory, f"{target}-{force_field_name}")
                protonated_pdb = Path(
                    target_dir,
                    "setup",
                    f"{target}-{force_field_name}-protonated.pdb",
                )
                minimized_pdb = Path(
                    target_dir,
                    "setup",
                    f"{target}-{force_field_name}-minimized.pdb",
                )
                assign_parameters(
                    simulation_platform="openmm",
                    nonbonded_cutoff=NONBONDED_CUTOFF,
                    vdw_switch_width=VDW_SWITCH_WIDTH,
                    protonated_pdb_file=str(protonated_pdb),
                    solvated_pdb_file=str(minimized_pdb),
                    parametrized_system=str(query_system_path),
                    water_model=force_fields[query_ff]["water_model"],
                    force_field_file=force_fields[query_ff]["force_field_file"],
                    water_model_file=force_fields[query_ff]["water_model_file"],
                )

        query_result = reweight_reference(query_system_paths)

        for target_index, target in enumerate(target_list):
            chi_square = query_result[0][target_index]
            chi_square_uncertainty = query_result[1][target_index]
            effective_samples = query_result[2][target_index]
            effective_samples_uncertainty = query_result[3][target_index]

            print(
                f"{query_ff:33s} {target:14s} {chi_square:9.4f} "
                f"({chi_square_uncertainty:7.4f}) "
                f"{int(numpy.round(effective_samples)):6d} "
                f"({effective_samples_uncertainty:5.1f})"
            )

        # Write reweighting potential for each target
        for target_index, target in enumerate(target_list):
            output_prefix = Path(
                output_directory,
                f"{target}-{force_field_name}-{query_ff}",
            )

            numpy.savetxt(
                f"{output_prefix}-reweighting-potential.dat",
                reweight_reference.reweighting_potential[target_index],
            )
            numpy.savetxt(
                f"{output_prefix}-uncorrelated-reweighting-potential.dat",
                reweight_reference.uncorrelated_reweighting_potential[target_index],
            )


if __name__ == "__main__":
    main()
