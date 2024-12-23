from pathlib import Path

import click
import numpy
import pandas
import pymbar
from openmm import unit
from pymbar import timeseries


def run_mbar(
    reduced_bias_potentials: numpy.typing.ArrayLike,
    reduced_query_potential: numpy.typing.ArrayLike,
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
    reduced_query_potential
        Array with shape (N_total_samples) of the reduced reweighting potential
        used to evaluate the free energy surface.
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
    mbar_fes.generate_fes(
        reduced_query_potential,
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
    unbiased_weights = numpy.exp(-reduced_query_potential) / Z
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
    "-n",
    "--nmr-fit-directory",
    type=click.STRING,
    default="gaussian-force-fields",
    show_default=True,
    help="Directory containing reweighting potentials from NMR fits.",
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
    "-q",
    "--query-force-field",
    type=click.STRING,
    default="null-0.0.3-pair-nmr-1e5-opc3",
    show_default=True,
    help="Name of force field used to compute the free energy surface.",
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
    nmr_fit_directory,
    output_directory,
    query_force_field,
    replica_to_analyze,
    target,
    bin_width,
):
    umbrella_directory = Path(
        output_directory,
        f"{target}-{force_field}",
    )
    analysis_directory = Path(umbrella_directory, "analysis")

    temperature = 298.0 * unit.kelvin
    RT = unit.MOLAR_GAS_CONSTANT_R * temperature
    beta = 1.0 / RT.value_in_unit(unit.kilocalorie_per_mole)

    # Read time series of collective variable and sample indices for correlated
    # and uncorrelated samples
    if replica_to_analyze is None:
        mbar_samples_path = Path(
            analysis_directory,
            f"{target}-{force_field}-mbar-samples.dat",
        )
        mbar_uncorrelated_samples_path = Path(
            analysis_directory,
            f"{target}-{force_field}-mbar-uncorrelated-samples.dat",
        )
    else:
        mbar_samples_path = Path(
            analysis_directory,
            f"{target}-{force_field}-mbar-{replica_to_analyze}-samples.dat",
        )
        mbar_uncorrelated_samples_path = Path(
            analysis_directory,
            f"{target}-{force_field}-mbar-{replica_to_analyze}-uncorrelated-"
                "samples.dat",
        )

    mbar_samples_df = pandas.read_csv(
        mbar_samples_path,
        index_col=0,
    )
    mbar_uncorrelated_samples_df = pandas.read_csv(
        mbar_uncorrelated_samples_path,
        index_col=0,
    )

    dihedral_cv = mbar_samples_df["Dihedral CV"].values
    uncorrelated_dihedral_cv = mbar_uncorrelated_samples_df["Dihedral CV"].values

    # Read umbrella energy constants, window centers, and number of correlated
    # and uncorrelated samples per window
    windows = mbar_samples_df["Window"].unique()
    N_windows = len(windows)

    window_centers = numpy.zeros(N_windows)
    umbrella_energy_constants = numpy.zeros(N_windows)
    N_samples = numpy.zeros(N_windows, dtype=int)
    N_uncorrelated_samples = numpy.zeros(N_windows, dtype=int)

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

        umbrella_energy_constants[window_index] = umbrella_energy_constant
        window_centers[window_index] = window_center

        N_samples[window_index] = len(
            mbar_samples_df[mbar_samples_df["Window"] == window]
        )
        N_uncorrelated_samples[window_index] = len(
            mbar_uncorrelated_samples_df[
                mbar_uncorrelated_samples_df["Window"] == window
            ]
        )

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

    # Read reweighting potential for query force field
    reweighting_potential = numpy.loadtxt(Path(
        nmr_fit_directory,
        f"{query_force_field}-reweighting-potential.dat",
    ))
    uncorrelated_reweighting_potential = numpy.loadtxt(Path(
        nmr_fit_directory,
        f"{query_force_field}-uncorrelated-reweighting-potential.dat",
    ))

    # Get centers of bins at which to evaluate the unbiased free energy surface
    min_x = numpy.floor(dihedral_cv.min() / bin_width) * bin_width
    max_x = numpy.ceil(dihedral_cv.max() / bin_width) * bin_width
    bin_centers = numpy.arange(min_x + bin_width / 2, max_x, bin_width)

    # Run MBAR to get window offsets, the unbiased free energy surface, MBAR
    # weight denominators for reweighting from the mixture distribution, and the
    # number of effective samples for reweighitng to the unbiased state
    window_offsets, unbiased_free_energy_surface, Z, N_eff = run_mbar(
        reduced_bias_potentials,
        reweighting_potential,
        N_samples,
        dihedral_cv,
        bin_centers,
        bin_width,
    )

    # Read bootstrap sample indices from resampling the uncorrelated sample
    # indices with replacemet
    bootstrap_sample_indices = mbar_uncorrelated_samples_df.loc[
        :,
        mbar_uncorrelated_samples_df.columns.str.startswith(
            "Bootstrap Sample Indices"
        ),
    ].values.T
    bootstrap_samples = bootstrap_sample_indices.shape[0]

    # Run MBAR for each bootstrap sample
    bootstrap_unbiased_free_energy_surface = numpy.zeros(
        (bootstrap_samples, len(bin_centers))
    )

    for bootstrap_index, resampled_indices in enumerate(bootstrap_sample_indices):
        bootstrap_mbar_output = run_mbar(
            uncorrelated_reduced_bias_potentials[:, resampled_indices],
            uncorrelated_reweighting_potential[resampled_indices],
            N_uncorrelated_samples,
            uncorrelated_dihedral_cv[resampled_indices],
            bin_centers,
            bin_width,
            initial_window_offsets=window_offsets,
        )

        bootstrap_unbiased_free_energy_surface[bootstrap_index] = bootstrap_mbar_output[
            1
        ]

    # Get uncertainties in the unbiased free energy surface from the bootstrap
    # estimates
    unbiased_free_energy_surface_uncertainties = (
        bootstrap_unbiased_free_energy_surface.std(axis=0, ddof=1)
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

    free_energy_df = pandas.DataFrame(
        {
            "Bin Center": bin_centers,
            "Free Energy (kcal mol^-1)": unbiased_free_energy_surface / beta,
            "Free Energy Uncertainty (kcal mol^-1)": (
                unbiased_free_energy_surface_uncertainties / beta
            ),
        }
    )
    free_energy_df.to_csv(f"{out_prefix}-{query_force_field}-free-energy.dat")


if __name__ == "__main__":
    main()
