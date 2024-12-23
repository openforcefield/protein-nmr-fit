from pathlib import Path

import click
import numpy
import pandas
import pymbar
from openmm import unit
from proteinbenchmark import benchmark_targets
from pymbar import timeseries


def run_mbar(
    reduced_bias_potentials: numpy.typing.ArrayLike,
    N_samples: numpy.typing.ArrayLike,
    collective_variable_samples: numpy.typing.ArrayLike,
    bin_centers: numpy.typing.ArrayLike,
    kde_bandwidth: float,
    initial_window_offsets: numpy.typing.ArrayLike | None = None,
) -> tuple[
    numpy.typing.NDArray[numpy.float64],
    numpy.typing.NDArray[numpy.float64],
    numpy.typing.NDArray[numpy.float64],
    int,
]:
    """
    Run MBAR to get window offsets, the unbiased free energy surface, MBAR
    weight denominators for reweighting from the mixture distribution, and the
    number of effective samples for reweighitng to the unbiased state.

    Parameters
    ----------
    reduced_bias_potentials
        Array with shape (N_windows, N_total_samples) where
        reduced_bias_potentials[i][j] is the bias potential for window i
        evaluated at sample j.
    N_samples
        Array with shape (N_windows,) of the number of samples per window.
    collective_variable_samples
        Array with shape (N_total_samples,) of the collective variable values.
    bin_centers
        The centers of the bins at which to evaluate the unbiased free energy
        surface.
    kde_bandwidth
        The bandwidth of the kernel density estimator of the free energy
        surface.
    initial_window_offests
        Array with shape (N_windows,) of an initial guess for the window offsets
        fed to the MBAR optimizer.

    Returns
    -------
    window_offsets
        Array with shape (N_windows,) of free energy offsets between windows.
    unbiased_free_energy_surface
        Array with shape (len(bin_centers),) of the estimate of the unbiased
        free energy surface from a Gaussian kernel density estimator.
    Z
        Array with shape (N_total_samples,) of denominators of the MBAR weights
        for reweighting from the mixture distribution.
    N_eff
        Number of effective samples for reweighting from the mixture
        distribution to the unbiased state.
    """

    # Set up MBAR to estimate window offsets and free energy surface
    mbar_fes = pymbar.FES(
        reduced_bias_potentials,
        N_samples,
        mbar_options={"initial_f_k": initial_window_offsets},
    )

    # Get free energy offsets between umbrella windows
    mbar_free_energies = mbar_fes.get_mbar().compute_free_energy_differences(
        compute_uncertainty=False
    )
    min_window_index = mbar_free_energies["Delta_f"][0].argmin()
    window_offsets = mbar_free_energies["Delta_f"][min_window_index]

    # Generate the unbiased free energy surface from a kernel density estimator
    unbiased_potentials = numpy.zeros(len(collective_variable_samples))
    mbar_fes.generate_fes(
        unbiased_potentials,
        collective_variable_samples,
        fes_type="kde",
        kde_parameters={"bandwidth": kde_bandwidth},
    )

    unbiased_free_energy_surface = mbar_fes.get_fes(
        bin_centers,
        reference_point="from-lowest",
    )["f_i"]

    # Denominator of MBAR weights for sampled windows
    Z = numpy.sum(
        N_samples * numpy.exp(window_offsets - reduced_bias_potentials.T),
        axis=1,
    )

    # Get unbiased weights. Same as
    # unbiased_weights = pymbar.MBAR(
    #     numpy.vstack([reduced_bias_potentials, unbiased_potentials]),
    #     numpy.array(list(N_samples) + [0])
    # ).weights()[:, -1]
    unbiased_weights = numpy.exp(-unbiased_potentials) / Z
    unbiased_weights /= unbiased_weights.sum()

    # Number of effective samples after reweighting to unbiased state. Same as
    # N_eff = pymbar.MBAR(
    #     numpy.vstack([reduced_bias_potentials, unbiased_potentials]),
    #     numpy.array(list(N_samples) + [0])
    # ).compute_effective_sample_number()
    N_eff = 1.0 / numpy.square(unbiased_weights).sum()

    return (window_offsets, unbiased_free_energy_surface, Z, N_eff)


@click.command()
@click.option(
    "-b",
    "--bootstrap-samples",
    type=click.INT,
    default=100,
    show_default=True,
    help="Number of bootstrap samples to generate for uncertainty estimates.",
)
@click.option(
    "-c",
    "--cumulative-directory",
    type=click.STRING,
    default="reweight-scalar-couplings",
    show_default=True,
    help="Directory containing reweighting potentials for cumulative force "
        "fields.",
)
@click.option(
    "-e",
    "--end-length",
    type=click.INT,
    default=None,
    show_default=True,
    help="Number of frames to read from the end of the trajectory. Default is "
        "all frames.",
)
@click.option(
    "-f",
    "--force-fields",
    type=click.STRING,
    default="null-0.0.3-pair-nmr-1e5-opc3,null-0.0.3-pair-opc3",
    show_default=True,
    help="Comma-separated list of force field names used to sample "
        "trajectories.",
)
@click.option(
    "-l",
    "--length",
    type=click.INT,
    default=None,
    show_default=True,
    help="Number of frames to read per trajectory. Default is all frames.",
)
@click.option(
    "-o",
    "--output_directory",
    type=click.STRING,
    default="results",
    show_default=True,
    help="Directory containing umbrella simulation output.",
)
@click.option(
    "-r",
    "--replica-to-analyze",
    type=click.STRING,
    default=None,
    show_default=True,
    help="Replica index to analyze. Default is to use all replicas.",
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
    "--bin-width",
    type=click.FLOAT,
    default=0.002,
    show_default=True,
    help="Width of histogram bins for free energy surface.",
)
def main(
    bootstrap_samples,
    cumulative_directory,
    end_length,
    force_fields,
    length,
    output_directory,
    replica_to_analyze,
    target,
    bin_width,
):
    mbar_str = "mbar"
    if "-0.7-" in force_fields:
        mbar_str = f"{mbar_str}-0.7"
    elif "-0.8-" in force_fields:
        mbar_str = f"{mbar_str}-0.8"
    if replica_to_analyze is not None:
        mbar_str = f"{mbar_str}-{replica_to_analyze}"

    force_fields = force_fields.strip("'\"").split(",")

    temperature = benchmark_targets[target]["temperature"].to_openmm()
    RT = unit.MOLAR_GAS_CONSTANT_R * temperature
    beta = 1.0 / RT.value_in_unit(unit.kilocalorie_per_mole)

    # Read MBAR samples for all force fields
    N_windows = list()
    window_centers = list()
    umbrella_energy_constants = list()

    N_samples = list()
    fraction_native_contacts = list()
    sample_force_fields = list()
    sample_replicas = list()
    sample_windows = list()

    N_uncorrelated_samples = list()
    uncorrelated_fraction_native_contacts = list()
    uncorrelated_sample_force_fields = list()
    uncorrelated_sample_replicas = list()
    uncorrelated_sample_windows = list()
    uncorrelated_sample_indices = list()

    bootstrap_sample_indices = list()

    for force_field in force_fields:
        umbrella_directory = Path(output_directory, f"{target}-{force_field}")
        analysis_directory = Path(umbrella_directory, "analysis")

        # Read time series of collective variable for correlated and
        # uncorrelated samples
        mbar_samples_path = Path(
            analysis_directory,
            f"{target}-{force_field}-{mbar_str}-samples.dat",
        )
        mbar_uncorrelated_samples_path = Path(
            analysis_directory,
            f"{target}-{force_field}-{mbar_str}-uncorrelated-samples.dat",
        )

        mbar_samples_df = pandas.read_csv(
            mbar_samples_path,
            index_col=0,
        )
        mbar_uncorrelated_samples_df = pandas.read_csv(
            mbar_uncorrelated_samples_path,
            index_col=0,
        )

        fraction_native_contacts.extend(
            mbar_samples_df["Fraction Native Contacts"].values
        )
        sample_force_fields.extend([force_field] * mbar_samples_df.shape[0])
        sample_replicas.extend(mbar_samples_df["Replica"].values)
        sample_windows.extend(mbar_samples_df["Window"].values)

        uncorrelated_fraction_native_contacts.extend(
            mbar_uncorrelated_samples_df["Fraction Native Contacts"].values
        )
        uncorrelated_sample_force_fields.extend(
            [force_field] * mbar_uncorrelated_samples_df.shape[0]
        )
        uncorrelated_sample_replicas.extend(
            mbar_uncorrelated_samples_df["Replica"].values
        )
        uncorrelated_sample_windows.extend(
            mbar_uncorrelated_samples_df["Window"].values
        )

        # Read sample indices for uncorrelated samples
        uncorrelated_sample_indices.extend(
            mbar_uncorrelated_samples_df["Indices"].values
        )

        # Read bootstrap sample indices from resampling the uncorrelated sample
        # indices with replacement
        bootstrap_sample_indices.extend(
            mbar_uncorrelated_samples_df.loc[
                :,
                mbar_uncorrelated_samples_df.columns.str.startswith(
                    "Bootstrap Sample Indices"
                ),
            ].values + sum(N_uncorrelated_samples)
        )

        # Read umbrella energy constants, window centers, and number of correlated
        # and uncorrelated samples per window
        windows = mbar_samples_df["Window"].unique()
        N_windows.append(len(windows))

        for window_index, window in enumerate(windows):
            window_out_files = Path(umbrella_directory, "replica-1").glob(
                f"{target}-{force_field}-1-{window:02d}-*.out"
            )
            with open(next(window_out_files), "r") as out_file:
                for line in out_file:
                    fields = line.split()
                    if fields[0] == "umbrella_energy_constant":
                        umbrella_energy_constant = float(fields[1]) * beta
                    elif fields[0] == "window_center":
                        window_center = float(fields[1])

            umbrella_energy_constants.append(umbrella_energy_constant)
            window_centers.append(window_center)

            N_samples.append(
                len(mbar_samples_df[mbar_samples_df["Window"] == window])
            )
            N_uncorrelated_samples.append(
                len(
                    mbar_uncorrelated_samples_df[
                        mbar_uncorrelated_samples_df["Window"] == window
                    ]
                )
            )

    umbrella_energy_constants = numpy.array(umbrella_energy_constants)
    window_centers = numpy.array(window_centers)
    N_samples = numpy.array(N_samples, dtype=int)
    fraction_native_contacts = numpy.array(fraction_native_contacts)
    N_uncorrelated_samples = numpy.array(N_uncorrelated_samples, dtype=int)
    uncorrelated_fraction_native_contacts = numpy.array(
        uncorrelated_fraction_native_contacts
    )
    bootstrap_sample_indices = numpy.array(bootstrap_sample_indices).T

    N_total_samples = N_samples.sum()
    N_total_uncorrelated_samples = N_uncorrelated_samples.sum()

    # Evaluate the bias potentials for samples from all windows
    reduced_bias_potentials = (
        umbrella_energy_constants[:, numpy.newaxis]
        * numpy.square(
            fraction_native_contacts - window_centers[:, numpy.newaxis]
        )
    )
    uncorrelated_reduced_bias_potentials = (
        umbrella_energy_constants[:, numpy.newaxis]
        * numpy.square(
            uncorrelated_fraction_native_contacts
            - window_centers[:, numpy.newaxis]
        )
    )

    # Add potential differences between force fields
    column_window_index = 0
    sample_index = 0
    uncorrelated_sample_index = 0
    for column_index in range(len(force_fields)):
        column_N_windows = N_windows[column_index]
        column_N_samples = N_samples[
            column_window_index : column_window_index + column_N_windows
        ].sum()
        column_N_uncorrelated_samples = N_uncorrelated_samples[
            column_window_index : column_window_index + column_N_windows
        ].sum()

        row_window_index = 0

        for row_index in range(len(force_fields)):
            row_N_windows = N_windows[row_index]

            if row_index != column_index:
                reweighting_potential_path = Path(
                    cumulative_directory,
                    f"{target}-{force_fields[column_index]}-"
                        f"{force_fields[row_index]}-reweighting-potential.dat",
                )
                reweighting_potential = numpy.loadtxt(reweighting_potential_path)

                reduced_bias_potentials[
                    row_window_index : row_window_index + row_N_windows,
                    sample_index : sample_index + column_N_samples
                ] += reweighting_potential

                uncorrelated_reweighting_potential_path = Path(
                    cumulative_directory,
                    f"{target}-{force_fields[column_index]}-"
                        f"{force_fields[row_index]}-uncorrelated-reweighting-"
                        "potential.dat",
                )
                uncorrelated_reweighting_potential = numpy.loadtxt(
                    uncorrelated_reweighting_potential_path
                )

                uncorrelated_reduced_bias_potentials[
                    row_window_index : row_window_index + row_N_windows,
                    uncorrelated_sample_index
                        : uncorrelated_sample_index + column_N_uncorrelated_samples
                ] += uncorrelated_reweighting_potential

            row_window_index += row_N_windows

        column_window_index += column_N_windows
        sample_index += column_N_samples
        uncorrelated_sample_index += column_N_uncorrelated_samples

    # Get centers of bins at which to evaluate the unbiased free energy surface
    min_x = numpy.floor(fraction_native_contacts.min() / bin_width) * bin_width
    max_x = numpy.ceil(fraction_native_contacts.max() / bin_width) * bin_width
    bin_centers = numpy.arange(min_x + bin_width / 2, max_x, bin_width)

    # Run MBAR to get window offsets, the unbiased free energy surface, MBAR
    # weight denominators for reweighting from the mixture distribution, and the
    # number of effective samples for reweighitng to the unbiased state
    window_offsets, unbiased_free_energy_surface, Z, N_eff = run_mbar(
        reduced_bias_potentials,
        N_samples,
        fraction_native_contacts,
        bin_centers,
        bin_width,
    )

    # Compute overlap matrix
    mbar_weights = (
        numpy.exp(window_offsets - reduced_bias_potentials.T)
        / Z[:, numpy.newaxis]
    )
    overlap_matrix = N_samples * (mbar_weights.T @ mbar_weights)

    print("\nWindow Overlap")
    for window_index in range(N_windows[0]):
        out_str = f"    {window_index:2d}"
        window_offset = window_index
        for ff_windows in N_windows[1:]:
            overlap = overlap_matrix[window_offset, window_offset + ff_windows]
            out_str += f" {overlap:10.8f}"
            window_offset += ff_windows
        print(out_str)

    # Run MBAR for each bootstrap sample
    bootstrap_samples = bootstrap_sample_indices.shape[0]
    bootstrap_window_offsets = numpy.zeros((bootstrap_samples, sum(N_windows)))
    bootstrap_unbiased_free_energy_surface = numpy.zeros(
        (bootstrap_samples, len(bin_centers))
    )
    bootstrap_N_eff = numpy.zeros(bootstrap_samples, dtype=int)

    uncorrelated_sample_dict = {
        "Force Field": uncorrelated_sample_force_fields,
        "Replica": uncorrelated_sample_replicas,
        "Window": uncorrelated_sample_windows,
        "Indices": uncorrelated_sample_indices,
        "Fraction Native Contacts": uncorrelated_fraction_native_contacts,
    }

    for bootstrap_index, resampled_indices in enumerate(bootstrap_sample_indices):
        bootstrap_mbar_output = run_mbar(
            uncorrelated_reduced_bias_potentials[:, resampled_indices],
            N_uncorrelated_samples,
            uncorrelated_fraction_native_contacts[resampled_indices],
            bin_centers,
            bin_width,
            initial_window_offsets=window_offsets,
        )

        bootstrap_window_offsets[bootstrap_index] = bootstrap_mbar_output[0]
        bootstrap_unbiased_free_energy_surface[bootstrap_index] = bootstrap_mbar_output[
            1
        ]
        bootstrap_N_eff[bootstrap_index] = bootstrap_mbar_output[3]

        uncorrelated_sample_dict[f"Bootstrap Sample Indices {bootstrap_index}"] = (
            resampled_indices
        )
        uncorrelated_sample_dict[f"MBAR Weight Denominator {bootstrap_index}"] = (
            bootstrap_mbar_output[2]
        )

    # Get uncertainties in window offsets, the unbiased free energy surface,
    # and the number of effective samples from the bootstrap estimates
    window_offset_uncertainties = bootstrap_window_offsets.std(axis=0, ddof=1)
    unbiased_free_energy_surface_uncertainties = (
        bootstrap_unbiased_free_energy_surface.std(axis=0, ddof=1)
    )
    N_eff_uncertainty = bootstrap_N_eff.std(axis=0, ddof=1)
    print(
        f"\nNumber of effective unbiased samples: {N_eff:.2f} +/- "
        f"{N_eff_uncertainty:.2f}"
    )

    # Write MBAR output using pandas
    umbrella_directory = Path(output_directory, f"{target}-{force_fields[0]}")
    analysis_directory = Path(umbrella_directory, "analysis")
    out_prefix = Path(analysis_directory, f"{target}-{force_fields[0]}-mbar-cum")
    if replica_to_analyze != None:
        out_prefix = f"{out_prefix}-{replica_to_analyze}"
    if length != None:
        time = int(numpy.round(length / 10))
        out_prefix = f"{out_prefix}-{time}ns"
    if end_length != None:
        time = int(numpy.round(end_length / 10))
        out_prefix = f"{out_prefix}-last-{time}ns"

    window_df = pandas.DataFrame(
        {
            "N_samples": N_samples,
            "Window Offsets": window_offsets,
            "Window Offset Uncertainties": window_offset_uncertainties,
        }
    )
    window_df.to_csv(f"{out_prefix}-windows.dat")

    free_energy_df = pandas.DataFrame(
        {
            "Bin Center": bin_centers,
            "Free Energy (kcal mol^-1)": unbiased_free_energy_surface / beta,
            "Free Energy Uncertainty (kcal mol^-1)": (
                unbiased_free_energy_surface_uncertainties / beta
            ),
        }
    )
    free_energy_df.to_csv(f"{out_prefix}-free-energy.dat")

    sample_df = pandas.DataFrame(
        {
            "Force Field": sample_force_fields,
            "Replica": sample_replicas,
            "Window": sample_windows,
            "Fraction Native Contacts": fraction_native_contacts,
            "MBAR Weight Denominator": Z,
        }
    )
    sample_df.to_csv(f"{out_prefix}-samples.dat")

    pandas.DataFrame(uncorrelated_sample_dict).to_csv(
        f"{out_prefix}-uncorrelated-samples.dat"
    )

    # Construct weights from free energy differences
    # mbar_weights = mbar_fes.get_mbar().weights()
    # mbar_weights = numpy.exp(window_offsets - reduced_bias_potentials.T)
    # mbar_weights /= numpy.sum(N_samples * mbar_weights, axis=1)[:, None]

    # Compute expectations of observables in the unbiased state
    # expectation_result = mbar_fes.get_mbar().compute_expectations(
    #     fraction_native_contacts,
    #     uncertainty_method="bootstrap",
    # )
    # expectation = expectation_result["mu"]
    # expectation_uncertainty = expectation_result["sigma"]
    # expectation = numpy.sum(unbiased_weights * fraction_native_contacts)


if __name__ == "__main__":
    main()
