from pathlib import Path

import click
import numpy
import pandas
import pymbar
from openmm import unit
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
    "--force-field",
    type=click.STRING,
    default="null-0.0.3-pair-opc3",
    show_default=True,
    help="Name of force field used to sample the trajectory.",
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
    default="butane",
    show_default=True,
    help="Name of benchmark target.",
)
@click.option(
    "-w",
    "--bin-width",
    type=click.FLOAT,
    default=numpy.deg2rad(1.5),
    show_default=True,
    help="Width of histogram bins for free energy surface.",
)
def main(
    bootstrap_samples,
    end_length,
    force_field,
    length,
    output_directory,
    replica_to_analyze,
    target,
    bin_width,
):
    umbrella_directory = Path(
        output_directory,
        f"{target}-{force_field}",
    )
    analysis_directory = Path(umbrella_directory, "analysis")

    N_replicas = 3
    if replica_to_analyze is None:
        replicas = [str(replica) for replica in numpy.arange(1, N_replicas + 1)]
    else:
        replicas = [replica_to_analyze]

    N_windows = 24
    windows = [f"{i:02d}" for i in range(N_windows)]

    temperature = 298.0 * unit.kelvin
    RT = unit.MOLAR_GAS_CONSTANT_R * temperature
    beta = 1.0 / RT.value_in_unit(unit.kilocalorie_per_mole)

    # Read umbrella energy constants, window centers, and time series of
    # collective variable, then subsample the latter to get uncorrelated samples
    window_centers = numpy.zeros(N_windows)
    umbrella_energy_constants = numpy.zeros(N_windows)

    N_samples = numpy.zeros(N_windows, dtype=int)
    dihedral_cv = list()
    sample_replicas = list()
    sample_windows = list()

    N_uncorrelated_samples = numpy.zeros(N_windows, dtype=int)
    uncorrelated_dihedral_cv = list()
    uncorrelated_sample_replicas = list()
    uncorrelated_sample_windows = list()
    uncorrelated_sample_indices = list()

    for window_index, window in enumerate(windows):
        # Read umbrella energy constant and window center
        window_out_files = Path(umbrella_directory, "replica-1").glob(
            f"{target}-{force_field}-1-{window}-*.out"
        )
        with open(next(window_out_files), "r") as out_file:
            for line in out_file:
                fields = line.split()
                if fields[0] == "umbrella_energy_constant":
                    umbrella_energy_constant = float(fields[1]) * beta
                elif fields[0] == "window_center":
                    window_center = float(fields[1])

        umbrella_energy_constants[window_index] = umbrella_energy_constant
        window_centers[window_index] = window_center

        N_window_samples = 0
        N_window_uncorrelated_samples = 0

        for replica in replicas:
            # Read time series of collective variable
            window_cv_path = Path(
                umbrella_directory,
                f"replica-{replica}",
                f"window-{window}",
                f"{target}-{force_field}-production-fraction-native-contacts.dat",
            )
            window_dihedral_cv = numpy.loadtxt(
                window_cv_path,
                usecols=1,
                max_rows=length,
            )

            if end_length != None:
                window_dihedral_cv = (
                    window_dihedral_cv[-end_length:]
                )

            N_window_samples += len(window_dihedral_cv)
            dihedral_cv.extend(window_dihedral_cv)
            sample_replicas.extend([replica] * len(window_dihedral_cv))

            # Subsample collective variable time series to get uncorrelated
            # samples
            statistical_inefficiency = timeseries.statistical_inefficiency(
                window_dihedral_cv
            )
            window_uncorrelated_sample_indices = timeseries.subsample_correlated_data(
                window_dihedral_cv,
                g=statistical_inefficiency,
            )

            N_window_uncorrelated_samples += len(window_uncorrelated_sample_indices)
            uncorrelated_dihedral_cv.extend(
                window_dihedral_cv[window_uncorrelated_sample_indices]
            )
            uncorrelated_sample_replicas.extend(
                [replica] * len(window_uncorrelated_sample_indices)
            )
            uncorrelated_sample_indices.extend(window_uncorrelated_sample_indices)

        sample_windows.extend([window] * N_window_samples)
        uncorrelated_sample_windows.extend([window] * N_window_uncorrelated_samples)

        N_uncorrelated_samples[window_index] = N_window_uncorrelated_samples
        N_samples[window_index] = N_window_samples

    dihedral_cv = numpy.array(dihedral_cv)
    uncorrelated_dihedral_cv = numpy.array(uncorrelated_dihedral_cv)

    N_total_samples = N_samples.sum()
    N_total_uncorrelated_samples = N_uncorrelated_samples.sum()

    # Evaluate the bias potentials for samples from all windows
    delta_phi = numpy.abs(dihedral_cv - window_centers[:, numpy.newaxis])
    reduced_bias_potentials = (
        umbrella_energy_constants[:, numpy.newaxis]
        * numpy.square(numpy.min([delta_phi, 2 * numpy.pi - delta_phi], axis=0))
    )

    uncorrelated_delta_phi = numpy.abs(
        uncorrelated_dihedral_cv - window_centers[:, numpy.newaxis]
    )
    uncorrelated_reduced_bias_potentials = (
        umbrella_energy_constants[:, numpy.newaxis] * numpy.square(
            numpy.min(
                [uncorrelated_delta_phi, 2 * numpy.pi - uncorrelated_delta_phi],
                axis=0,
            )
        )
    )

    # Get centers of bins at which to evaluate the unbiased free energy surface
    min_x = numpy.floor(dihedral_cv.min() / bin_width) * bin_width
    max_x = numpy.ceil(dihedral_cv.max() / bin_width) * bin_width
    bin_centers = numpy.arange(min_x + bin_width / 2, max_x, bin_width)

    # Run MBAR to get window offsets, the unbiased free energy surface, MBAR
    # weight denominators for reweighting from the mixture distribution, and the
    # number of effective samples for reweighitng to the unbiased state
    window_offsets, unbiased_free_energy_surface, Z, N_eff = run_mbar(
        reduced_bias_potentials,
        N_samples,
        dihedral_cv,
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
    for window_index in range(N_windows - 1):
        print(
            f"    {window_index:2d} "
            f"{overlap_matrix[window_index, window_index + 1]:10.8f}"
        )

    # Get bootstrap sample indices by resampling the uncorrelated sample indices
    # with replacement
    rng = numpy.random.default_rng()

    if bootstrap_samples > 0:
        bootstrap_sample_indices = numpy.zeros(
            (bootstrap_samples, N_total_uncorrelated_samples),
            dtype=int,
        )

        # Take the same number of samples from each window in each bootstrap
        end_sample_index = 0
        for window_index in range(N_windows):
            start_sample_index = end_sample_index
            end_sample_index += N_uncorrelated_samples[window_index]
            window_sample_indices = numpy.arange(
                start_sample_index,
                end_sample_index,
            )
            resampled_indices = rng.integers(
                N_uncorrelated_samples[window_index],
                size=(bootstrap_samples, N_uncorrelated_samples[window_index]),
            )
            bootstrap_sample_indices[:, window_sample_indices] = window_sample_indices[
                resampled_indices
            ]

    # Run MBAR for each bootstrap sample
    bootstrap_window_offsets = numpy.zeros((bootstrap_samples, N_windows))
    bootstrap_unbiased_free_energy_surface = numpy.zeros(
        (bootstrap_samples, len(bin_centers))
    )
    bootstrap_N_eff = numpy.zeros(bootstrap_samples, dtype=int)

    uncorrelated_sample_dict = {
        "Replica": uncorrelated_sample_replicas,
        "Window": uncorrelated_sample_windows,
        "Indices": uncorrelated_sample_indices,
        "Dihedral CV": uncorrelated_dihedral_cv,
    }

    for bootstrap_index, resampled_indices in enumerate(bootstrap_sample_indices):
        bootstrap_mbar_output = run_mbar(
            uncorrelated_reduced_bias_potentials[:, resampled_indices],
            N_uncorrelated_samples,
            uncorrelated_dihedral_cv[resampled_indices],
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
    out_prefix = str(Path(analysis_directory, f"{target}-{force_field}-mbar"))
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
            "Window": windows,
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
            "Replica": sample_replicas,
            "Window": sample_windows,
            "Dihedral CV": dihedral_cv,
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
    #     dihedral_cv,
    #     uncertainty_method="bootstrap",
    # )
    # expectation = expectation_result["mu"]
    # expectation_uncertainty = expectation_result["sigma"]
    # expectation = numpy.sum(unbiased_weights * dihedral_cv)


if __name__ == "__main__":
    main()
