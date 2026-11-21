# NuQuLib numerical figures and tables

This dataset accompanies arXiv:2607.08047. Notebook 3 reproduces Figs. 3 and 6–14 and Tables II and IV–VIII from the supplied data. Figure 7 has two panels and Fig. 8 three, giving 13 PDF files for the manuscript. The cost script also writes the earlier four-panel layout of the fixed-accuracy panels and its separate ratio figure. Tables I and III of the manuscript are written by hand. Figure and table numbers follow the revised manuscript. Each notebook runs independently on a CPU and uses local inputs after the dependencies are installed.

## Run

From this directory, set up Python 3.12 and start JupyterLab.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `03_figures_tables.ipynb` and select **Restart Kernel and Run All Cells**. It creates `build/` for the figure PDFs, six LaTeX tables and the QPE cost report.

| Notebook | Calculation |
| --- | --- |
| `01_counting_grouping.ipynb` | Recounts Pauli strings, computes QWC, FC and Majorana measurement settings, and checks pairwise commutator sums for three small spaces. |
| `02_spectral_tutorial.ipynb` | Reconstructs the QPE distributions for $^3$n and $^6$He (Figs. 10 and 11) and checks all 18 product-formula errors for Fig. 13. |
| `03_figures_tables.ipynb` | Generates the numerical figures and tables from the counts, Hamiltonian summaries, sector matrices and simulation results in `data/`. |

For Figs. 9, 12 and 14, the plotting notebook uses stored angular-momentum weights, spectra and VQE histories. For Table VII, it computes success counts and confidence bounds from the 7,200 supplied energy estimates. [The data guide](data/README.md) gives the units, model spaces and figure-to-file mapping.

## Optional structural counts

To run the Julia counter in notebook 1, install Julia 1.12 and run `julia --project=lib -e 'using Pkg; Pkg.instantiate()'` from this directory. Set `run_julia = True` in the final cell. The default examples are SDPFSDG (124 qubits, 3,764,359 structural strings) and NN+3N at $e_{\max}=1$ (51,377 structural strings). Set `full_sweep = True` for all structural rows.

Run `shasum -a 256 -c SHA256SUMS` (macOS) or `sha256sum -c SHA256SUMS` (Linux) from `data/` to verify the inputs. NuQuLib-derived code is covered by [its MIT license](lib/NuQuLib-LICENSE).
