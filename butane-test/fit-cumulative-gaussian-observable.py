from pathlib import Path
from typing import TypedDict

import click
import numpy
import pandas
import scipy
from openff.toolkit import ForceField, Topology
from openff.units import unit
from openmm import unit as openmm_unit


class TorsionDict(TypedDict):
    dihedral_name: str
    smirks: str
    periodicity: list[int]
    phase: list[float]


def test_gradient(function_and_gradient, x, args=tuple()):
    """
    Test function gradient using finite difference approximation.
    """

    if not isinstance(args, tuple):
        args = (args,)

    value, gradient = function_and_gradient(x, *args)
    gradient_approximation = scipy.optimize.approx_fprime(
        x, lambda x: function_and_gradient(x, *args)[0]
    )
    sse = numpy.mean(numpy.square(gradient_approximation - gradient))
    print(value)
    print(numpy.sqrt(sse))
    print(gradient)
    print(gradient_approximation)


class NMRFitter:
    """
    Optimize force field parameter targeting NMR observables by reweighting.
    """

    def __init__(
        self,
        result_directory: str,
        fit_parameters: TorsionDict,
        force_field: ForceField,
        force_field_name: str,
        observable_list: list[str],
        target_list: list[str],
        truncate_observables: bool,
        replica: int | None = None,
    ):
        """
        Setup the time series of estimated observables and basis functions for
        the reweighting potential.

        Parameters
        ----------
        result_directory
            Top-level directory containing simulation results.
        fit_parameters
            Dictiondary of dihedral name, SMIRKS pattern, periodicities, and
            phases for fit parameters.
        force_field
            OpenFF ForceField object representing the force field used to
            parametrize the molecule.
        force_field_name
            Name of the force field used to sample the observable time series.
        observable_list
            List of observable names.
        target_list
            List of target names.
        truncate_observables
            Truncate experimental scalar couplings to extrema of the Karplus
            curve.
        replica
            Replica index to read. Default is to use all replicas.
        """

        self.N_parameters = sum(
            [len(parameter["phase"]) for parameter in fit_parameters.values()]
        )

        # Sample indices and MBAR weight denominators by target
        mbar_samples = list()
        mbar_uncorrelated_samples = list()

        for target in target_list:
            target_directory = Path(
                result_directory,
                f"{target}-{force_field_name}",
                "analysis",
            )

            if replica is None:
                mbar_samples_path = Path(
                    target_directory,
                    f"{target}-{force_field_name}-mbar-cum-samples.dat",
                )
                mbar_uncorrelated_samples_path = Path(
                    target_directory,
                    f"{target}-{force_field_name}-mbar-cum-uncorrelated-samples.dat",
                )
            else:
                mbar_samples_path = Path(
                    target_directory,
                    f"{target}-{force_field_name}-mbar-cum-{replica}-samples.dat",
                )
                mbar_uncorrelated_samples_path = Path(
                    target_directory,
                    f"{target}-{force_field_name}-mbar-cum-{replica}-uncorrelated-"
                        "samples.dat",
                )

            # Read correlated sample indices and MBAR weight denominator
            target_samples_df = pandas.read_csv(
                mbar_samples_path,
                index_col=0,
                usecols=lambda column: column != "Dihedral CV",
            )

            # Read uncorrelated sample indices and MBAR weight denominators
            target_uncorrelated_samples_df = pandas.read_csv(
                mbar_uncorrelated_samples_path,
                index_col=0,
                usecols=lambda column: column != "Dihedral CV",
            )

            mbar_samples.append(target_samples_df)
            mbar_uncorrelated_samples.append(target_uncorrelated_samples_df)

        self.get_target_betas(target_list)

        self.set_up_observables(
            result_directory,
            force_field_name,
            observable_list,
            target_list,
            truncate_observables,
            mbar_samples,
            mbar_uncorrelated_samples,
        )

        self.set_up_reweighting_basis_functions(
            result_directory,
            fit_parameters,
            force_field,
            force_field_name,
            target_list,
            mbar_samples,
            mbar_uncorrelated_samples,
        )

        N_targets = len(target_list)
        self.reweighting_potential = [None for _ in range(N_targets)]
        self.uncorrelated_reweighting_potential = [None for _ in range(N_targets)]
        self.chi_square = numpy.zeros(N_targets)
        self.chi_square_uncertainty = numpy.zeros(N_targets)
        self.effective_samples = numpy.zeros(N_targets)
        self.effective_samples_uncertainty = numpy.zeros(N_targets)

    def __call__(
        self,
        parameters: numpy.typing.ArrayLike,
        alpha: float,
        observables_to_skip: list[str] | None = None,
    ):
        """
        Compute the value of the regularized loss function and its gradient with
        respect to the fit parameters.

        Parameters
        ----------
        parameters
            numpy array containing the values of the fit parameters.
        alpha
            Regularization strength
        observables_to_skip
            List of observable types to skip for evaluation of the objective
            function. Used for leave-one-out cross validation.
        """

        return self.compute_loss_function_and_gradient(
            parameters, alpha, observables_to_skip=observables_to_skip
        )

    def get_target_betas(self, target_list: list[str]):
        """
        Read ensemble temperature for observable targets and compute
        thermodynamic beta, i.e. (k_B T)^-1.

        Parameters
        ----------
        target_list
            List of target names.
        """

        target_betas = list()
        for target in target_list:
            target_temperature = 298.0 * openmm_unit.kelvin
            RT = openmm_unit.MOLAR_GAS_CONSTANT_R * target_temperature
            target_betas.append(
                1.0 / RT.value_in_unit(openmm_unit.kilocalorie_per_mole)
            )

        self.target_betas = target_betas

    def set_up_observables(
        self,
        result_directory: str,
        force_field_name: str,
        observable_list: list[str],
        target_list: list[str],
        truncate_observables: bool,
        mbar_samples: list[pandas.DataFrame],
        mbar_uncorrelated_samples: list[pandas.DataFrame],
    ):
        """
        Read the uncorrelated sample indices and weight denominator from MBAR
        and the time series of estimated observables.

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
        mbar_samples
            Correlated sample indices and MBAR weight denominators by target.
        mbar_uncorrelated_samples
            Uncorrelated sample indices and MBAR weight denominators by target.
        """

        # List of columns to read for observable DataFrames
        experiment_column = (
            "Truncated Experiment" if truncate_observables else "Experiment"
        )
        df_columns = [
            "Frame",
            "Observable",
            experiment_column,
            "Experiment Uncertainty",
            "Computed",
        ]

        # Observable types, experimental values, experimental uncertainties, and
        # unweighted estimates of observables by target
        self.observable_types = list()
        self.experimental_observables = list()
        self.experimental_variances = list()

        # Sample estimates of observables by target
        self.sampled_observables = list()
        self.uncorrelated_observables = list()

        # MBAR weight denominators for correlated samples
        self.mbar_weight_denominators = list()

        # Bootstrap indices and MBAR weight denominators for bootstrap samples
        self.bootstrap_sample_indices = list()
        self.bootstrap_mbar_weight_denominators = list()

        print(
            "Reference_Force_Field             Target         Chi^2     "
            "(StDev)   N_eff  (StDev)"
        )

        for target_index, target in enumerate(target_list):
            # DataFrame of correlated and uncorrelated sample indices for this
            # target
            target_samples_df = mbar_samples[target_index]
            target_uncorrelated_samples_df = mbar_uncorrelated_samples[target_index]

            # Read observables for uncorrelated sample indices in each window
            target_observable_df = pandas.DataFrame()
            target_uncorrelated_observable_df = pandas.DataFrame()

            for force_field in target_samples_df["Force Field"].unique():
                target_ff_directory = Path(
                    result_directory,
                    f"{target}-{force_field}",
                    "analysis",
                )
                force_field_samples_df = target_samples_df[
                    target_samples_df["Force Field"] == force_field
                ]

                for window in force_field_samples_df["Window"].unique():
                    window_samples_df = force_field_samples_df[
                        force_field_samples_df["Window"] == window
                    ]

                    for replica in window_samples_df["Replica"].unique():
                        window_uncorrelated_sample_indices = (
                            target_uncorrelated_samples_df.loc[
                                (target_uncorrelated_samples_df["Force Field"] == force_field)
                                & (target_uncorrelated_samples_df["Replica"] == replica)
                                & (target_uncorrelated_samples_df["Window"] == window),
                                "Indices",
                            ].values
                        )

                        observable_path = Path(
                            target_ff_directory,
                            f"{target}-{force_field}-{replica}-{window:02d}-"
                            "gaussian-observable-time-series.dat",
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
                "Truncated Experiment",
                "Experiment Uncertainty",
            ]
            target_observable_df.set_index(index_columns, inplace=True)
            target_observable_df.sort_index(inplace=True)
            target_uncorrelated_observable_df.set_index(index_columns, inplace=True)
            target_uncorrelated_observable_df.sort_index(inplace=True)
            observable_groups = target_observable_df.index.unique()

            N_observables = observable_groups.size
            N_samples = target_samples_df.shape[0]
            N_uncorrelated_samples = target_uncorrelated_samples_df.shape[0]

            target_observable_types = list()
            target_experimental_observables = numpy.zeros(N_observables)
            target_experimental_uncertainties = numpy.zeros(N_observables)
            target_sampled_observables = numpy.zeros((N_observables, N_samples))
            target_uncorrelated_observables = numpy.zeros(
                (N_observables, N_uncorrelated_samples)
            )

            for observable_index, observable_group in enumerate(observable_groups):
                target_observable_types.append(observable_group[0])
                target_experimental_observables[observable_index] = observable_group[1]
                target_experimental_uncertainties[observable_index] = observable_group[
                    2
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

            self.observable_types.append(target_observable_types)
            self.experimental_observables.append(target_experimental_observables)
            self.experimental_variances.append(
                numpy.square(target_experimental_uncertainties)
            )
            self.sampled_observables.append(target_sampled_observables)
            self.uncorrelated_observables.append(target_uncorrelated_observables)

            # Get MBAR weight denominators for correlated samples
            target_mbar_weight_denominators = target_samples_df[
                "MBAR Weight Denominator"
            ].values
            self.mbar_weight_denominators.append(target_mbar_weight_denominators)

            # Get bootstrap indices and MBAR weight denominators for bootstrap
            # samples
            target_bootstrap_sample_indices = target_uncorrelated_samples_df.loc[
                :,
                target_uncorrelated_samples_df.columns.str.startswith(
                    "Bootstrap Sample Indices"
                ),
            ].values
            target_bootstrap_mbar_weight_denominators = (
                target_uncorrelated_samples_df.loc[
                    :,
                    target_uncorrelated_samples_df.columns.str.startswith(
                        "MBAR Weight Denominator"
                    ),
                ].values
            )
            self.bootstrap_sample_indices.append(target_bootstrap_sample_indices)
            self.bootstrap_mbar_weight_denominators.append(
                target_bootstrap_mbar_weight_denominators
            )

            # Get chi^2 value and number of effective samples for unbiased state
            mbar_weights = 1.0 / target_mbar_weight_denominators
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

            # Get uncertainties in chi^2 and number of effective samples from
            # bootstrapping over uncorrelated samples
            bootstrap_mbar_weights = 1.0 / target_bootstrap_mbar_weight_denominators
            bootstrap_mbar_weights /= bootstrap_mbar_weights.sum(axis=0)

            bootstrap_observables = target_uncorrelated_observables[
                :,
                target_bootstrap_sample_indices,
            ]
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
                f"{force_field_name:33s} {target:14s} "
                f"{reference_chi_square:9.4f} ({chi_square_uncertainty:7.4}) "
                f"{reference_effective_samples:6d} "
                f"({effective_samples_uncertainty:5.1f})"
            )

    def set_up_reweighting_basis_functions(
        self,
        result_directory: str,
        fit_parameters: TorsionDict,
        force_field: ForceField,
        force_field_name: str,
        target_list: list[str],
        mbar_samples: list[pandas.DataFrame],
        mbar_uncorrelated_samples: list[pandas.DataFrame],
    ):
        """
        Construct the time series of basis functions for the reweighting
        potential from the time series of dihedral angles.

        Parameters
        ----------
        result_directory
            Top-level directory containing simulation results.
        fit_parameters
            Dictiondary of dihedral name, SMIRKS pattern, periodicities, and
            phases for fit parameters.
        force_field
            OpenFF ForceField object representing the force field used to
            parametrize the molecule.
        force_field_name
            Name of the force field used to sample the observable time series.
        target_list
            List of target names.
        mbar_samples
            Correlated sample indices and MBAR weight denominators by target.
        mbar_uncorrelated_samples
            Uncorrelated sample indices and MBAR weight denominators by target.
        """

        self.N_parameters = sum(
            [len(parameter["phase"]) for parameter in fit_parameters.values()]
        )

        # Basis functions for reweighting potential
        # Phi_p(t) = sum_{i in dihedrals_p} 1 + cos(n_i phi_i(t) - gamma_i)
        # such that U_rw(k, t) = \sum_p k_p Phi_p(t)
        self.reweighting_basis_functions = list()
        self.uncorrelated_basis_functions = list()

        for target_index, target in enumerate(target_list):
            # DataFrame of correlated and uncorrelated sample indices for this
            # target
            target_samples_df = mbar_samples[target_index]
            target_uncorrelated_samples_df = mbar_uncorrelated_samples[target_index]

            # Construct a (N_parameters, N_samples) numpy array of basis
            # functions for this target
            N_samples = target_samples_df.shape[0]
            N_uncorrelated_samples = target_uncorrelated_samples_df.shape[0]

            target_basis_functions = numpy.zeros((self.N_parameters, N_samples))
            target_uncorrelated_basis_functions = numpy.zeros(
                (self.N_parameters, N_uncorrelated_samples)
            )

            total_sample_index = 0
            total_uncorrelated_sample_index = 0

            # Read dihedral angles for uncorrelated sample indices in each window
            for target_force_field in target_samples_df["Force Field"].unique():
                target_ff_directory = Path(
                    result_directory,
                    f"{target}-{target_force_field}",
                    "analysis",
                )

                force_field_samples_df = target_samples_df[
                    target_samples_df["Force Field"] == target_force_field
                ]

                for window in force_field_samples_df["Window"].unique():
                    window_samples_df = force_field_samples_df[
                        force_field_samples_df["Window"] == window
                    ]

                    for replica in window_samples_df["Replica"].unique():
                        window_uncorrelated_sample_indices = (
                            target_uncorrelated_samples_df.loc[
                                (target_uncorrelated_samples_df["Force Field"] == target_force_field)
                                & (target_uncorrelated_samples_df["Replica"] == replica)
                                & (target_uncorrelated_samples_df["Window"] == window),
                                "Indices",
                            ].values
                        )

                        dihedral_path = Path(
                            target_ff_directory,
                            f"{target}-{target_force_field}-{replica}-{window:02d}-"
                            "dihedrals.dat",
                        )
                        dihedral_df = pandas.read_csv(
                            dihedral_path,
                            usecols=["Frame", "Dihedral Name", "Dihedral (deg)"],
                            index_col=1,
                        )
                        dihedral_df.sort_index(inplace=True)

                        # Uncorrelated samples
                        uncorrelated_dihedral_df = dihedral_df[
                            dihedral_df["Frame"].isin(window_uncorrelated_sample_indices)
                        ]

                        sample_start_index = total_sample_index
                        uncorrelated_sample_start_index = total_uncorrelated_sample_index
                        total_sample_index += dihedral_df.loc[dihedral_df.index[0]].shape[0]
                        total_uncorrelated_sample_index += uncorrelated_dihedral_df.loc[
                            uncorrelated_dihedral_df.index[0]
                        ].shape[0]

                        # Loop over fit parameters
                        parameter_index = 0
                        for parameter_id, fit_parameter in fit_parameters.items():
                            dihedral_name = fit_parameter["dihedral_name"]

                            dihedral_time_series = numpy.deg2rad(
                                dihedral_df.loc[
                                    dihedral_name,
                                    "Dihedral (deg)",
                                ].values
                            )
                            dihedral_uncorrelated_time_series = numpy.deg2rad(
                                uncorrelated_dihedral_df.loc[
                                    dihedral_name,
                                    "Dihedral (deg)",
                                ].values
                            )

                            # Loop over Fourier terms for this fit parameter
                            fourier_index = parameter_index

                            for periodicity, phase in zip(
                                fit_parameter["periodicity"],
                                fit_parameter["phase"],
                            ):
                                target_basis_functions[
                                    fourier_index,
                                    sample_start_index:total_sample_index,
                                ] += 1 + numpy.cos(
                                    periodicity * dihedral_time_series - phase
                                )

                                target_uncorrelated_basis_functions[
                                    fourier_index,
                                    uncorrelated_sample_start_index:total_uncorrelated_sample_index,
                                ] += 1 + numpy.cos(
                                    periodicity * dihedral_uncorrelated_time_series
                                    - phase
                                )

                                fourier_index += 1

                            parameter_index += len(fit_parameter["phase"])

            self.reweighting_basis_functions.append(target_basis_functions)
            self.uncorrelated_basis_functions.append(
                target_uncorrelated_basis_functions
            )

    def compute_loss_function_and_gradient(
        self,
        parameters: numpy.typing.ArrayLike,
        alpha: float,
        observables_to_skip: list[str] = None,
    ):
        """
        Compute the value of the regularized loss function and its gradient with
        respect to the fit parameters.

        Parameters
        ----------
        parameters
            numpy array containing the values of the fit parameters.
        alpha
            Regularization strength.
        observables_to_skip
            List of observable types to skip for evaluation of the objective
            function. Used for leave-one-out cross validation.
        """
        if len(parameters) != self.N_parameters:
            raise ValueError(
                "Length of parameter vector to evaluate loss function "
                f"({len(parameters)}) does not match length of fit parameter "
                f"vector ({self.N_parameters})"
            )

        total_chi_square = 0
        total_chi_square_gradient = numpy.zeros(self.N_parameters)

        for target_index, target_basis_functions in enumerate(
            self.reweighting_basis_functions
        ):
            beta = self.target_betas[target_index]
            target_experimental_observables = self.experimental_observables[
                target_index
            ]
            target_experimental_variances = self.experimental_variances[target_index]
            target_sampled_observables = self.sampled_observables[target_index]
            target_uncorrelated_observables = self.uncorrelated_observables[
                target_index
            ]
            target_mbar_weight_denominators = self.mbar_weight_denominators[
                target_index
            ]
            target_bootstrap_sample_indices = self.bootstrap_sample_indices[
                target_index
            ]
            target_bootstrap_mbar_weight_denominators = (
                self.bootstrap_mbar_weight_denominators[target_index]
            )
            target_uncorrelated_basis_functions = self.uncorrelated_basis_functions[
                target_index
            ]

            if observables_to_skip is not None:
                observable_mask = [
                    observable not in observables_to_skip
                    for observable in self.observable_types[target_index]
                ]
                target_experimental_observables = target_experimental_observables[
                    observable_mask
                ]
                target_experimental_variances = target_experimental_variances[
                    observable_mask
                ]
                target_sampled_observables = target_sampled_observables[observable_mask]
                target_uncorrelated_observables = target_uncorrelated_observables[
                    observable_mask
                ]

            # U_rw(k, t) = sum_i k_i * (1 + cos(n_i * phi_i(t) - gamma_i)
            reweighting_potential = beta * numpy.dot(
                parameters,
                target_basis_functions,
            )
            uncorrelated_reweighting_potential = beta * numpy.dot(
                parameters,
                target_uncorrelated_basis_functions,
            )

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
                numpy.exp(-reweighting_potential) / target_mbar_weight_denominators
            )

            # W(k, t) = exp(-U_rw(k, t)) / Z(t) / ( sum_t exp(-U_rw(k, t)) / Z(t) )
            # N_eff(k) = (sum_t W(k, t))^2 / sum_t W(k, t)^2
            mbar_weight_normalization = mbar_weights.sum()
            if mbar_weight_normalization == 0.0:
                mbar_weights = numpy.zeros(mbar_weights.size)
                self.effective_samples[target_index] = 0.0
            else:
                mbar_weights = mbar_weights / mbar_weight_normalization
                self.effective_samples[target_index] = (
                    1.0 / numpy.square(mbar_weights).sum()
                )

            # < O_j >(k) = sum_t W(k, t) * O_j(t)
            reweighted_estimates = numpy.sum(
                mbar_weights * target_sampled_observables,
                axis=1,
            )

            # < dU_rw / dk_i >(k)
            #     = sum_t W(k, t) * (1 + cos(n_i * phi_i(t) - gamma_i))
            reweighting_potential_derivatives = numpy.sum(
                mbar_weights * target_basis_functions,
                axis=1,
            )

            # < O_j * dU_rw / dk_i>(k)
            #     = sum_t W(k, t) * O_j(t) * (1 + cos(n_i * phi_i(t) - gamma_i))
            # Building up the (N_parameters, N_observables, N_samples) array and
            # then calling numpy.sum(axis=2) is faster but requires more memory
            observable_derivative_product = numpy.zeros(
                (self.N_parameters, target_experimental_observables.size)
            )
            for parameter_index in range(self.N_parameters):
                observable_derivative_product[parameter_index] = numpy.sum(
                    target_basis_functions[parameter_index]
                    * target_sampled_observables
                    * mbar_weights,
                    axis=1,
                )

            # d< O_j > / dk_i = beta *
            #     (< O_j > * < dU_rw / dk_i > - < O_j * dU_rw / dk_i >)
            # Multiply by beta once at the end
            reweighted_estimate_derivatives = beta * (
                reweighting_potential_derivatives[:, numpy.newaxis]
                * reweighted_estimates
                - observable_derivative_product
            )

            # chi^2(k) = sum_j (< O_j >(k) - O_j,exp )^2 / sigma_j,exp^2
            target_errors = reweighted_estimates - target_experimental_observables
            weighted_target_errors = target_errors / target_experimental_variances
            chi_square = numpy.sum(weighted_target_errors * target_errors)
            self.chi_square[target_index] = chi_square
            total_chi_square += chi_square

            # dchi^2 / dk_i = 2 *
            #     sum_j (< O_j >(k) - O_j,exp) / sigma_j,exp^2 * d< O_j > / dk_i
            # Multiply by 2 once at the end
            total_chi_square_gradient += numpy.sum(
                reweighted_estimate_derivatives * weighted_target_errors,
                axis=1,
            )

            # Get uncertainties in chi^2 and number of effective samples from
            # bootstrapping over uncorrelated samples
            uncorrelated_boltzmann_factors = numpy.exp(
                -uncorrelated_reweighting_potential
            )
            bootstrap_mbar_weights = (
                uncorrelated_boltzmann_factors[target_bootstrap_sample_indices]
                / target_bootstrap_mbar_weight_denominators
            )

            bootstrap_weight_normalization = bootstrap_mbar_weights.sum(axis=0)
            if numpy.any(bootstrap_weight_normalization == 0.0):
                bootstrap_mbar_weights = numpy.zeros(bootstrap_mbar_weights.shape)
                bootstrap_effective_samples = numpy.zeros(
                    target_bootstrap_sample_indices.shape[1]
                )
            else:
                bootstrap_mbar_weights = (
                    bootstrap_mbar_weights / bootstrap_weight_normalization
                )
                bootstrap_effective_samples = 1.0 / numpy.square(
                    bootstrap_mbar_weights
                ).sum(axis=0)

            bootstrap_observables = target_uncorrelated_observables[
                :,
                target_bootstrap_sample_indices,
            ]
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

            self.chi_square_uncertainty[target_index] = bootstrap_chi_square.std(ddof=1)
            self.effective_samples_uncertainty[target_index] = (
                bootstrap_effective_samples.std(ddof=1)
            )

        # L(k) = chi^2(k) + alpha sum_i k_i^2
        loss_function = total_chi_square + alpha * numpy.sum(numpy.square(parameters))
        # dL / dk_i = dchi^2 / dk_i + 2 alpha k_i
        loss_gradient = 2 * (total_chi_square_gradient + alpha * parameters)

        return loss_function, loss_gradient


@click.command()
@click.option(
    "-a",
    "--alpha",
    default=None,
    show_default=True,
    type=click.FLOAT,
    help="Strength of L2 regularization term. Default is to do leave-one-out "
    "cross validation to choose the optimal value using a line search.",
)
@click.option(
    "-c",
    "--chi-square-only",
    is_flag=True,
    default=False,
    help="Evaluate the loss function once with no parameter optimization.",
)
@click.option(
    "-f",
    "--force-field-name",
    default="null-0.0.3-pair-opc3",
    show_default=True,
    type=click.STRING,
    help="Name of force field to optimize.",
)
@click.option(
    "-g",
    "--gradient-test",
    is_flag=True,
    default=False,
    help="Test the gradient of the loss function using a finite difference method.",
)
@click.option(
    "-i",
    "--input-directory",
    default="results",
    show_default=True,
    type=click.STRING,
    help="Directory path containing MBAR analysis and time series of observables.",
)
@click.option(
    "-l",
    "--log10-alpha",
    default=None,
    show_default=True,
    type=click.FLOAT,
    help="Logarithmic strength of L2 regularization term. Default is to do "
    "leave-one-out cross validation to choose the optimal value using a "
    "line search.",
)
@click.option(
    "-o",
    "--output-prefix",
    default="final-force-field",
    show_default=True,
    type=click.STRING,
    help="Prefix for file path of output files.",
)
@click.option(
    "-r",
    "--replica",
    default=None,
    show_default=True,
    type=click.STRING,
    help="Replica index to read. Default is to use all replicas.",
)
@click.option(
    "-t/-e",
    "--truncate-observables/--experimental-observables",
    default=True,
    help="Truncate scalar couplings to the extrema of the Karplus curve.",
)
def main(
    alpha,
    chi_square_only,
    force_field_name,
    gradient_test,
    input_directory,
    log10_alpha,
    output_prefix,
    replica,
    truncate_observables,
):

    # List of parameter ids for parameters to be fit and their associated
    # protein dihedral
    fit_parameter_ids = {
        "t2": "C-C-C-C",
    }

    # List of observable types to train on
    observable_list = ["3j_c_c"]

    # List of training targets
    target_list = ["butane"]
    N_targets = len(target_list)

    # Get initial force field
    force_field = ForceField(
        Path("gaussian-force-fields", f"{force_field_name}.offxml")
    )

    # Get periodicities and phases for proper torsions associated with
    # parameters to be fit
    torsion_handler = force_field["ProperTorsions"]

    fit_parameters = dict()
    for parameter_id, dihedral_name in fit_parameter_ids.items():
        parameter = torsion_handler.get_parameter({"id": parameter_id})[0]
        fit_parameters[parameter_id] = {
            "dihedral_name": dihedral_name,
            "smirks": parameter.smirks,
            "periodicity": parameter.periodicity,
            "phase": [phase.to(unit.radian).m for phase in parameter.phase],
        }

    # Set up parameter optimization
    nmr_fitter = NMRFitter(
        input_directory,
        fit_parameters,
        force_field,
        force_field_name,
        observable_list,
        target_list,
        truncate_observables,
        replica=replica,
    )

    initial_guess = numpy.zeros(nmr_fitter.N_parameters)

    if gradient_test:
        # Test the gradient of the loss gradient by a finite difference method
        N_parameters = nmr_fitter.N_parameters
        test_gradient(
            nmr_fitter,
            numpy.arange(-N_parameters / 20 + 0.05, N_parameters / 20, 0.1),
            10.0,
        )

        return

    # Get number of observables of each type for all targets
    N_observables_by_type = dict()
    for observable in observable_list:
        N_observables_by_type[observable] = [
            len(
                [
                    observable_type
                    for observable_type in target_observable_types
                    if observable_type == observable
                ]
            )
            for target_observable_types in nmr_fitter.observable_types
        ]

    # Total number of observables
    N_observables_by_target = [
        target_experimental_observables.size
        for target_experimental_observables in nmr_fitter.experimental_observables
    ]

    if chi_square_only:
        # Print chi square value and number of effective samples for each
        # observable type and then for all observables
        print(
            "\nObservable Target         N_obs Chi^2     (StDev)   N_eff  (StDev)"
        )

        result = nmr_fitter(initial_guess, 0.0)
        for target_index, target in enumerate(target_list):
            reduced_chi_square = (
                nmr_fitter.chi_square[target_index]
                / N_observables_by_target[target_index]
            )

            print(
                f"Total      {target:14s} "
                f"{N_observables_by_target[target_index]:5d} "
                f"{reduced_chi_square:9.4f} "
                f"({nmr_fitter.chi_square_uncertainty[target_index]:7.4f}) "
                f"{int(numpy.round(nmr_fitter.effective_samples[target_index])):6d} "
                f"({nmr_fitter.effective_samples_uncertainty[target_index]:5.1f})"
            )

        return

    if alpha is not None:
        min_alpha = alpha
        print(f"Specified value of alpha: {min_alpha:9.3f}")

    elif log10_alpha is not None:
        min_alpha = numpy.power(10.0, log10_alpha)
        print(f"Specified value of alpha: {min_alpha:9.3f}")

    else:
        # Choose value of regularization weight alpha by cross validation
        # using a golden section line search
        def cross_validation_error(cv_log10_alpha):
            cv_alpha = numpy.power(10.0, cv_log10_alpha)

            N_observable_types = len(observable_list)
            cv_chi_square = numpy.zeros(N_observable_types)
            cv_chi_square_variances = numpy.zeros(N_observable_types)
            cv_effective_samples = numpy.zeros(N_observable_types)
            cv_effective_samples_variances = numpy.zeros(N_observable_types)

            for observable_index, observable in enumerate(observable_list):
                cv_training_observables = [
                    obs for obs in observable_list if obs != observable
                ]
                cv_test_observables = [observable]

                # Train on all observable types but one
                optimization_result = scipy.optimize.minimize(
                    nmr_fitter,
                    initial_guess,
                    args=(cv_alpha, cv_test_observables),
                    method="BFGS",
                    jac=True,
                )

                # Test on remaining observable type
                cv_chi_square[observable_index] = nmr_fitter(
                    optimization_result.x,
                    0.0,
                    observables_to_skip=cv_training_observables,
                )[0] / sum(N_observables_by_type[observable])

                cv_chi_square_variances[observable_index] = numpy.square(
                    nmr_fitter.chi_square_uncertainty
                ).sum() / numpy.square(sum(N_observables_by_type[observable]))

                cv_effective_samples[observable_index] = (
                    nmr_fitter.effective_samples.sum()
                )

                cv_effective_samples_variances[observable_index] = numpy.square(
                    nmr_fitter.effective_samples_uncertainty
                ).sum()

            cv_chi_square_mean = cv_chi_square.mean()
            cv_chi_square_uncertainty = (
                numpy.sqrt(cv_chi_square_variances.sum()) / cv_chi_square.size
            )

            cv_effective_samples_mean = cv_effective_samples.mean()
            cv_effective_samples_uncertainty = (
                numpy.sqrt(cv_effective_samples_variances.sum())
                / cv_effective_samples.size
            )

            print(
                f"{cv_log10_alpha:11.8f} {cv_chi_square_mean:9.4f}     "
                f"({cv_chi_square_uncertainty:7.4f}) "
                f"{cv_effective_samples_mean:6.0f}     "
                f"({cv_effective_samples_uncertainty:5.1f})"
            )

            return cv_chi_square_mean

        print("\nlog10_alpha CV_error_mean (StDev)   N_eff_mean (StDev)")

        for i in numpy.arange(8.0, -1.1, -1):
            cross_validation_error(i)
        cross_validation_error(-numpy.inf)
        return

        min_log10_alpha = scipy.optimize.golden(
            cross_validation_error,
            brack=(2.0, 2.5, 3.0),
        )
        min_alpha = numpy.power(10.0, min_log10_alpha)

        print(f"Optimal value of alpha: {min_alpha:9.3f}")

    # Run parameter optimization
    optimization_result = scipy.optimize.minimize(
        nmr_fitter,
        initial_guess,
        args=(min_alpha),
        method="BFGS",
        jac=True,
    )
    final_parameters = optimization_result.x

    # Print chi square value and number of effective samples for each
    # observable type and then for all observables
    print(
        "\nObservable Target         N_obs Init_Chi^2 (StDev)   Init_N_eff "
        "(StDev) Final_Chi^2 (StDev)   Final_N_eff (StDev)"
    )

    result = nmr_fitter(initial_guess, 0.0)
    initial_chi_square = numpy.array(nmr_fitter.chi_square)
    initial_chi_square_uncertainty = numpy.array(
        nmr_fitter.chi_square_uncertainty
    )
    initial_effective_samples = numpy.array(nmr_fitter.effective_samples)
    initial_effective_samples_uncertainty = numpy.array(
        nmr_fitter.effective_samples_uncertainty
    )

    result = nmr_fitter(final_parameters, 0.0)
    final_chi_square = numpy.array(nmr_fitter.chi_square)
    final_chi_square_uncertainty = numpy.array(
        nmr_fitter.chi_square_uncertainty
    )
    final_effective_samples = numpy.array(nmr_fitter.effective_samples)
    final_effective_samples_uncertainty = numpy.array(
        nmr_fitter.effective_samples_uncertainty
    )

    for target_index, target in enumerate(target_list):
        initial_reduced_chi_square = (
            initial_chi_square[target_index]
            / N_observables_by_target[target_index]
        )
        final_reduced_chi_square = (
            final_chi_square[target_index]
            / N_observables_by_target[target_index]
        )

        print(
            f"Total      {target:14s} "
            f"{N_observables_by_target[target_index]:5d} "
            f"{initial_reduced_chi_square:9.4f}  "
            f"({initial_chi_square_uncertainty[target_index]:7.4f}) "
            f"{int(numpy.round(initial_effective_samples[target_index])):6d}     "
            f"({initial_effective_samples_uncertainty[target_index]:5.1f}) "
            f"{final_reduced_chi_square:9.4f}   "
            f"({final_chi_square_uncertainty[target_index]:7.4f}) "
            f"{int(numpy.round(final_effective_samples[target_index])):6d}      "
            f"({final_effective_samples_uncertainty[target_index]:5.1f}) "
        )

    # Write final reweighting potential for each target for optimized parameters
    for target_index, target in enumerate(target_list):
        numpy.savetxt(
            f"{output_prefix}-reweighting-potential.dat",
            nmr_fitter.reweighting_potential[target_index],
        )
        numpy.savetxt(
            f"{output_prefix}-uncorrelated-reweighting-potential.dat",
            nmr_fitter.uncorrelated_reweighting_potential[target_index],
        )

    # Write optimized parameters to new force field OFFXML
    print("\nParameter_ID              n Phase(deg) Initial_k   Final_k")
    parameter_index = -1
    for parameter_id in fit_parameter_ids:
        fit_parameter = torsion_handler.get_parameter({"id": parameter_id})[0]
        for fourier_index in range(len(fit_parameter.periodicity)):
            parameter_index += 1
            periodicity = fit_parameter.periodicity[fourier_index]
            phase = fit_parameter.phase[fourier_index].m_as(unit.degree)
            initial_energy_constant = fit_parameter.k[fourier_index].m_as(
                unit.kilocalorie_per_mole
            )
            final_energy_constant = (
                initial_energy_constant + final_parameters[parameter_index]
            )
            print(
                f"{parameter_id:25s} {periodicity:1d}   {phase:8.3f} "
                f"{initial_energy_constant:12.8f} {final_energy_constant:12.8f}"
            )

            # Update force field parameter
            fit_parameter.k[fourier_index] = (
                final_energy_constant * unit.kilocalorie / unit.mole
            )

    force_field.to_file(f"{output_prefix}.offxml")


if __name__ == "__main__":
    main()
