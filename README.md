# Nuclear quantum algorithms · paper reproducibility
This repository accompanies the NuQuLib manuscript. It contains the inputs and notebooks for the numerical figures, tables and small verification calculations. The default notebooks run offline once the Python dependencies are installed. They use Python 3.12, a CPU, and only files in this checkout.

## Run
From this directory create a Python 3.12 environment. On macOS or Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open a notebook and select **Restart Kernel and Run All Cells**. Each notebook starts at this directory and can run on its own. To reproduce just the manuscript PDFs and tables, start with notebook 3.

| Notebook | What it does |
| --- | --- |
| `01_counting_grouping.ipynb` | Recomputes three small structural NN counts, QWC/FC/Majorana greedy settings from the retained Hamiltonians, and direct pairwise commutator sums. The final cell can call the original Julia counter for NN+3N or the entire structural sweep. |
| `02_spectral_tutorial.ipynb` | Reconstructs the exact finite-register QPE distributions for $^3$n and $^6$He, then recomputes the full six-qubit product-formula benchmark for Fig. 14. All calculations start from bundled matrices or coefficients. |
| `03_figures_tables.ipynb` | Recomputes the resource model, regenerates the 13 numerical PDFs for Figs. 2 and 6–14, copies the four schematics, and writes Tables I–VI in LaTeX. Output filenames match the numerical figures used by the manuscript. |

## Calculation coverage

| Item | Input to the default workflow |
| --- | --- |
| Fig. 2, Table I, Fig. 6(a,b) | Recorded matrix-element array sizes and complete structural count table; notebook 1 independently enumerates NN at $e_{\max}=0,1$ and the valence $p$ shell. |
| Table II | Small coefficient lists run through all three greedy algorithms. Published large-space reductions are calculated from the archived $M$ and $G$. An empty setting count is printed as a dash. |
| Table III | Manuscript analytical formulas. |
| Figs. 6(c,d), 7, 12 and Tables IV,V | Per-Hamiltonian commutator coefficients, norms and identity shifts. The notebook evaluates cost budgets, fits, normalizations and polynomial degrees. |
| Figs. 8, 9, 14 | Small Hamiltonians are directly usable for diagonalization and product-formula evolution. |
| Figs. 10, 11, 13 | Stored QKrylov/ODMD results, angular-momentum weights and VQE iteration histories feed the paper plotting code. |
| Table VI | The observed number of successes and its confidence bound are recomputed from the 7,200 stored estimates. The notebook uses the archived protocol settings and does not draw fresh validation samples. |

The small-operator archive also contains NN+3N at $e_{\max}=1$ (51,376 retained terms). Add its ID in notebook 1 to run all three groupings. Higher-cutoff NN+3N cases use stored summaries. For the optional original Julia structural counter, install Julia 1.12 and run `julia --project=lib -e 'using Pkg; Pkg.instantiate()'` once before enabling the cell. This setup is independent of the Python calculations.

## Spot checks against the manuscript

For the $p$ shell the independent structural counter gives 975 strings including the identity. The CK interaction retains 974 nonidentity terms. First fit gives 263 QWC, 47 FC and 48 Majorana settings. For NN+3N at $e_{\max}=1$ the Julia counter gives 51,377 structural strings, including 697 diagonal strings. The coefficient list contains 51,376 nonidentity terms; first fit gives 11,983 QWC, 1,004 FC and 1,807 Majorana settings.

## Data deposit
The `data/` directory is the independently archivable paper dataset. Its README describes the schemas, physics conventions, figure coverage and boundaries of recomputation. From `data/`, run `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux. A data-only ZIP contains that directory and can be extracted at the repository root. The Hamiltonians and summaries retain the archive's source identifiers. NuQuLib-derived code carries its upstream license in `lib/NuQuLib-LICENSE`.
