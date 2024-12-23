## Butane test workflow

(1) Data from the umbrella sampling run with force field $FF is located in results/butane-${FF}.
For the umbrella restraint U(Q) = k (Q - Q0)^2, the umbrella energy constants k and window centers Q0 are in results/butane-${FF}/replica-${REPLICA}/butane-${FF}-${REPLICA}-${WINDOW}-${JOBID}.out, and the time series of collective variables are in results/butane-${FF}/replica-${REPLICA}/window-${WINDOW}/butane-${FF}-production-fraction-native-contacts.dat.
The CV here is the C-C-C-C dihedral in butane.

(2) I obtained the unbiased free energy surface for this CV and the QM force field using python compute-mbar-free-energy.py -f null-0.0.3-pair-opc3, written to results/butane-${FF}/analysis/butane-${FF}-mbar-free-energy.dat.
The indices, CV values, and denominator of the MBAR weights for the unbiased state (Z in the script) are written to results/butane-${FF}/analysis/butane-${FF}-mbar-samples.dat.
The script also generates results/butane-${FF}/analysis/butane-${FF}-mbar-uncorrelated-samples.dat for the samples obtained from bootstrapping over the uncorrelated samples.

(3) To create a target observable for fitting, I created a Gaussian centered on 0 deg with a width of 15 deg, i.e. f(t) = exp(-(phi(t) / 15 deg)^2) where phi(t) is the butane C-C-C-C dihedral.
The loss function for the fit is L(k) = (<phi>(k) - 1)^2 + alpha * |k|^2 where <phi>(k) = sum_t W(k, t) * f(t) is the weighted average over the observable using the MBAR weights W(k, t) from the reweighting potential.
The first term is a least-squares error with a target value of 1 (so the error is minimized when samples with phi = 0 deg is favorable), and the second term is an L2 regularization with strength controlled by the hyperparameter alpha.

(4) I did a fit with alpha = 1E-2 using python fit-gaussian-observable.py -f null-0.0.3-pair-opc3 -l -2 -o gaussian-force-fields/null-0.0.3-pair-nmr-1e-2-opc3 > gaussian-force-fields/null-0.0.3-pair-nmr-1e-2-opc3.out.
The resulting force field is gaussian-force-fields/null-0.0.3-pair-nmr-1e-2-opc3.offxml, and the final values of the reweighting potential are written to gaussian-force-fields/null-0.0.3-pair-nmr-1e-2-opc3-reweighting-potential.dat and -uncorrelated-reweighting-potential.dat.

(5) I obtained the predicted free energy surface from this fit using python compute-reweighted-mbar-free-energy.py -f null-0.0.3-pair-opc3 -q null-0.0.3-pair-nmr-1e-2-opc3, written to results/butane-null-0.0.3-pair-opc3/analysis/butane-null-0.0.3-pair-opc3-mbar-null-0.0.3-pair-nmr-1e-2-opc3-free-energy.dat.
This uses the samples and MBAR weights from the QM force field but evaluates the free energy surface using the final reweighting potential from the fit rather than the unbiased state potential (i.e. an array of zeros).

(6) After running the umbrella sampling simulations, I obtained the new free energy surface using python compute-mbar-free-energy.py -f null-0.0.3-pair-nmr-1e-2-opc3.

(7) I obtained the plot of the free energy surface using python plot-free-energy.py -s 10 -l -e png, written to plots/butane-opc3-pooled-free-energy.png.
The solid blue line is results/butane-null-0.0.3-pair-opc3/analysis/butane-null-0.0.3-pair-opc3-mbar-free-energy.dat from (2), the dashed orange line is results/butane-null-0.0.3-pair-opc3/analysis/butane-null-0.0.3-pair-opc3-mbar-null-0.0.3-pair-nmr-1e-2-opc3-free-energy.dat from (5), and the solid orange line is results/butane-null-0.0.3-pair-nmr-1e-2-opc3/analysis/butane-null-0.0.3-pair-nmr-1e-2-opc3-mbar-free-energy.dat from (6).
