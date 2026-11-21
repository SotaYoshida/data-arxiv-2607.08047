# Numerical data

Energies and Pauli coefficients are in MeV. Keep this directory alongside the notebooks and `lib/` when reproducing the figures and tables. Figure and table numbers follow the revised manuscript.

| Files | Paper use and conventions |
| --- | --- |
| `structural_counts.tsv` | Table II and Fig. 7. Symmetry-structural $N_H$ includes the identity. The TSV records model cutoffs, the Julia source hash and the run environment. |
| `small_operators.npz` | Table V and direct checks of the commutator coefficients in Fig. 6 and Table IV. Four coefficient sets from the measurement benchmark. Each key is `case__field`. Supports are zero-based Majorana labels, padded with −1. Degree gives the active width. `coeffs_majorana` are real coefficients in original tie order; the identity has been removed. |
| `grouping.json` | Published retained-term and greedy setting counts (Table V). Missing setting counts are null or absent. The SDPFSDG row uses the same 124-qubit space as Table II and the valence panels of Figs. 7 and 8. |
| `operator_summaries.json` | Table IV, Figs. 6 and 8, and Table VIII. Retained term counts, interaction identifiers, commutator coefficients, identity shifts and coefficient one-norms. $C_{\rm pair}$ in MeV² is the once-per-anticommuting-pair sum. The identity is separate for the matched-error and QSVT models. The SDPFSDG row uses the mass-dependent interaction at $A=42$ ($Z=N=21$). Its input SHA-256 identifies the `.snt` file used for the largest valence space. |
| `nonzero_me.csv` | Fig. 3. Stored J/M-scheme array sizes, converted at eight bytes per element. |
| `qpe_sectors.json` | Figs. 10 and 11. Real sector matrices in the listed integer determinant basis, initial determinant bitmasks, energy shifts, and SHA-256 of the original interactions. |
| `demonstrations.json` | Figs. 9–12 and 14. Reference six-ancilla QPE bin energies/probabilities and exact levels with angular momenta (Figs. 10 and 11); the $^{20}$O Ritz history, two ODMD spectra and DOCI levels (Fig. 12); the $^{18}$O angular-momentum probabilities by cycle (Fig. 9); 500-iteration pUCCD energy and DOCI-error histories plus FCI energies for four nuclei (Fig. 14). These are numerical arrays exported from the manuscript tutorial calculations. |
| `trotter_order.json` | Fig. 13. Ordered nonidentity Pauli coefficients of the six-qubit CK Hamiltonian, benchmark settings, rotation counts, norm errors, fit parameters, versions and validation data. |
| `finite_shot.json`, `finite_shot_energies.npz` | Table VII. The six fixed protocols and all 1,200 energy estimates per protocol, indexed by nucleus and method. Success uses $\lvert E-E_0\rvert\leq0.100$ MeV. |
| `table_layouts.json` | LaTeX headers and labels for Tables IV (`coefficients`), V (`grouping`) and VII (`finite_shot`), and the analytical formulas of Table VI (`formulas`). |

The large-space inputs here are evaluated summaries. They allow reproduction of the paper figures and tables without repeating the underlying large NN+3N recoupling or grouping. The distribution includes neither the full large interaction matrices nor all large-space setting assignments. Figs. 9, 12 and 14 can be replotted from the numerical histories, which do not contain all stages needed to retrain or resimulate those examples.

## Simulation records

- **Fig. 9.** The first column is the initial state. Each later column follows a complete $J_z=0$ then $J_x=0$ projection cycle. Weights are normalized after successful postselection. The first entry of the $J=0$ row, $2.97\times10^{-6}$, is the initial $J=0$ weight, which sets the cumulative postselection probability.
- **Fig. 12.** The 15-dimensional zero-seniority $sd$-shell calculation uses 15 QKrylov basis sizes. The ODMD spectra use the two Hankel shapes recorded in the file, $5 \times 5$ and $10 \times 10$.
- **Fig. 14.** The four interactions in nuclear order are CKpot, USDB, GXPF1A and JUN45. The pUCCD trajectories use the Adam optimizer with step size 0.5 and $\beta_2=0.998$. The lower panels use the recorded DOCI differences, including the original floating-point residuals.
