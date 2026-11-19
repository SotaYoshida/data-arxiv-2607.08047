# Paper data
Energies and Pauli coefficients use MeV. The source snapshot corresponds to the revised manuscript supplied with the November 2026 handoff. All computational inputs for the default notebooks are in this directory. Extract a standalone data ZIP into the code repository so that this file is at `data/README.md`.

| Files | Paper use and conventions |
| --- | --- |
| `structural_counts.tsv` | Table I and Fig. 6(a,b). Symmetry-structural $N_H$ includes the identity. The TSV records model cutoffs, the Julia source hash and the run environment. |
| `small_operators.npz` | Table II and direct checks of Table IV. Four coefficient sets from the measurement benchmark. Each key is `case__field`. Supports are zero-based Majorana labels, padded with −1. Degree gives the active width. `coeffs_majorana` are real coefficients in original tie order; the identity has been removed. |
| `grouping.json` | Published retained-term and greedy setting counts. Missing setting counts are null or absent. The 124-qubit valence interaction and the 100-qubit structural model in Table I are distinct spaces. |
| `operator_summaries.json` | Retained Hamiltonian summaries, interaction identifiers, supplied provenance and coefficient moments. $C_{\rm pair}$ in MeV² is the once-per-anticommuting-pair sum. The identity is separate for the matched-error and QSVT models. The normalization-only 124-qubit row has `main_figure=false`. |
| `nonzero_me.csv` | Fig. 2. Stored J/M-scheme array sizes, converted at eight bytes per element. |
| `qpe_sectors.json` | Figs. 8 and 9. Real sector matrices in the listed integer determinant basis, initial determinant bitmasks, energy shifts, and SHA-256 of the original interactions. |
| `demonstrations.json` | Figs. 8–11 and 13. Reference six-ancilla QPE bin energies/probabilities and exact levels with angular momenta; the $^{20}$O Ritz history, two ODMD spectra and DOCI levels; the $^{18}$O angular-momentum probabilities by cycle; 500-iteration pUCCD energy and DOCI-error histories plus FCI energies for four nuclei. These are numerical arrays exported from the manuscript tutorial calculations. |
| `trotter_order.json` | Fig. 14. Ordered nonidentity Pauli coefficients of the six-qubit CK Hamiltonian, benchmark settings, rotation counts, norm errors, fit parameters, versions and validation data. |
| `finite_shot.json`, `finite_shot_energies.npz` | Table VI. The six fixed protocols and all 1,200 energy estimates per protocol, indexed by nucleus and method. Success uses $\lvert E-E_0\rvert\leq0.100$ MeV. |
| `table_layouts.json`, `artwork/` | LaTeX headers, labels, symbolic Table III, and the four original schematic PDFs (Figs. 1, 3–5). |

The large-space inputs here are evaluated summaries. They allow reproduction of the paper figures and tables without repeating the underlying large NN+3N recoupling or grouping. The distribution includes neither the full large interaction matrices nor all large-space setting assignments. Figs. 10, 11 and 13 can be replotted from the numerical histories, which do not contain all stages needed to retrain or resimulate those examples.

## Simulation records

- **Fig. 10.** The 15-dimensional zero-seniority $sd$-shell calculation uses 15 QKrylov basis sizes. The ODMD spectra use the two Hankel shapes recorded in the file, $5 \times 5$ and $10 \times 10$.
- **Fig. 11.** The first column is the initial state. Each later column follows a complete $J_z=0$ then $J_x=0$ projection cycle. Weights are normalized after successful postselection.
- **Fig. 13.** The four interactions in nuclear order are CKpot, USDB, GXPF1A and JUN45. The pUCCD trajectories use the Adam optimizer with step size 0.5 and $\beta_2=0.998$. The lower panels use the recorded DOCI differences, including the original floating-point residuals.

## Source snapshot

SHA-256 fingerprints of the manuscript-side sources used for these exports:

- `tutorial_NuQuLib.ipynb` · `427d2c9648166fe14c20b3ac9788acd51842f93b8fdd1998a2f4ccbe2ac48cf8`
- `drawing_Tcounts_trotter_error_NuQuLib_updated.ipynb` · `f0cb4069f0a5e8400f61e0358110abf59c924bda6be8270ef64eb43889bbf7cc`
- `exact_commutator_results.json` · `69f41aed5c155fce13bf5213c5a50987712354cb88399cab9bd03f607f059efc`
- `large_measurement_results.json` · `b49d66da0a2cd7dd2ea9a8ad2a5548206f93ac56b3d294bbfefb45b670457b9c`
- `measurement_grouping_settings.npz` · `faf2a9fbbe479a5d7d2ef9bd6cac88d6d2debd4d3006ecc6de3604a6af5d86f0`
- `trotter_order_benchmark.json` · `61e0f0041db5d272d002c8ec20a33ce7321acd9cb8631922c224061c1fc6fd91`
- `results.json` · `2b37f9349a97e5608ffe43ec63972a14799d7d18eb687001d6eb1f6220c883cc`
- `distributions.npz` · `90a906237de59ac0ae1011f68fc886617d5fb42d8404328a730ecfe463b7d397`

`SHA256SUMS` lists the data files, relative to this directory. The filenames refer to the contents, independently of the location where the archive was extracted.
