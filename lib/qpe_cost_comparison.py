"""QPE resource model and QSVT normalization table. Edit the settings, then run.

The input is the retained Pauli Hamiltonian summarized in the pair-results
JSON. Both methods remove its identity coefficient and restore it classically.
These are logical T-count estimates, conditional on an encoded eigenstate;
they do not include physical error correction or ground-state preparation.
"""
from pathlib import Path
import hashlib
import json
import math

SETTINGS = {
    "energy_errors_MeV": [1.0, 0.1, 0.001],
    "figure6_energy_MeV": 0.1,
    "phase_estimation_fraction": 0.5,
    "circuit_synthesis_norm_budget": 0.001,
    "qrom_tradeoff_q": 1,
    "shots": 1,
}

QSVT_SETTINGS = {
    "time_MeV_inverse": 0.25,
    "simulation_errors": [1e-7, 1e-4],
    "normalization_significant_digits": 3,
}


def normalization_report(rows, settings=None):
    """Use all verified coefficient sums, even without a commutator result.

    Degrees reproduce the manuscript's Lambert estimate and are checked
    against its factorial expression. They are estimates, not optimized
    QSVT phase sequences or a synthesis of a complete simulation circuit.
    """
    from scipy.special import lambertw
    settings = dict(QSVT_SETTINGS if settings is None else settings)
    time = settings["time_MeV_inverse"]
    assert time > 0
    result = []
    for row in rows:
        identity = row["identity_coefficient"]
        shifted = row["lambda_nonidentity"]
        norm = abs(identity) + shifted
        previous = row.get("lambda_H_including_identity_MeV")
        if previous is not None:
            assert math.isclose(norm, previous, rel_tol=1e-12)
        degrees = []
        tau = norm * time
        for epsilon in settings["simulation_errors"]:
            assert 0 < epsilon < 1
            argument = (math.log(8 / math.pi) - 2 * math.log(epsilon)) / (math.e * tau)
            estimate = tau / 2 * math.exp(float(lambertw(argument).real) + 1)
            degree = 2 * math.ceil(estimate / 2)
            log_error = lambda d: math.log(4) + d * math.log(tau / 2) - math.lgamma(d + 1)
            # Evaluate the factorial expression in log space for large degrees.
            while log_error(degree) > math.log(epsilon):
                degree += 2
            degrees.append({"epsilon_sim": epsilon, "lambert_estimate": estimate,
                            "even_degree": degree, "log_factorial_error_estimate": log_error(degree)})
        result.append({"id": row["id"], "nqubits": row["nqubits"],
                       "identity_coefficient_MeV": identity,
                       "lambda_nonidentity_MeV": shifted, "lambda_H_MeV": norm,
                       "degrees": degrees})
    return {"settings": settings,
            "normalization": "lambda_H = abs(identity coefficient) + lambda_nonidentity; no particle-number projection.",
            "degree_method": "Lambert estimate rounded up to an even integer, then checked against 4*(lambda_H*t/2)**d/d!; unrounded coefficient sums used.",
            "rows": result}


def normalization_table_tex(report):
    labels = {
        "valence_p": r"$p$ (CK)", "valence_sd": r"$sd$ (USDB)",
        "valence_psd": r"$psd$ (YSOX)", "valence_pf": r"$pf$ (GXPF1A)",
        "valence_sdpf_mu_A42": r"$sdpf$ (SDPF-MU)",
        "valence_sdpf_sdg_A42": r"$sd$--$pf$--$sdg$ (SDPFSDG)",
    }
    def scientific(value):
        digits = report["settings"]["normalization_significant_digits"]
        mantissa, power = f"{value:.{digits-1}e}".split("e")
        return rf"${mantissa}\times10^{{{int(power)}}}$"
    errors = [entry["epsilon_sim"] for entry in report["rows"][0]["degrees"]]
    headings = [rf"$10^{{{round(math.log10(epsilon))}}}$" for epsilon in errors]
    time = report["settings"]["time_MeV_inverse"]
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Hamiltonian coefficient one-norms, including identity, and estimated even polynomial degrees",
             rf"at $t={time:g}\,\mathrm{{MeV}}^{{-1}}$ for the indicated simulation errors.",
             r"Input and degree conventions are specified in Appendix~\ref{app:qsvt_time_evolution}.",
             r"\label{tab:Trotter_Error_QSVT}}", r"\begin{ruledtabular}",
             r"\begin{tabular}{lrrrr}",
             r" & & & \multicolumn{2}{c}{$d$ for $\epsilon_{\rm sim}$} \\",
             r"\cline{4-5}",
             r"Model space / interaction & $N_q$ & $\lambda_H$ (MeV) & " + " & ".join(headings) + r" \\",
             r"\hline"]
    for row in report["rows"]:
        name = row["id"]
        if name in ["nn_e0", "3nf_e1_compact_common"]:
            lines.append(r"\hline")
        if name.startswith("nn_e"):
            label = r"NN, $e_{\max}=" + name.split("_e")[1] + "$"
        elif name.startswith("3nf_e"):
            label = r"NN+3N, $e_{\max}=" + name.split("_e")[1].split("_")[0] + "$"
        else:
            label = labels[name]
        degrees = [f"{entry['even_degree']:,}" for entry in row["degrees"]]
        lines.append(" & ".join([label, str(row["nqubits"]), scientific(row["lambda_H_MeV"])] + degrees) + r" \\")
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def ceil_log2_integer(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("Expected a positive integer.")
    return (n - 1).bit_length()


def ceil_log2_positive(x):
    if not math.isfinite(x) or x <= 0:
        raise ValueError("Expected a finite positive number.")
    n = max(0, math.ceil(math.log2(x)))
    while 2**n < x:
        n += 1
    while n > 0 and 2**(n - 1) >= x:
        n -= 1
    return n


def qft_rotations(m):
    # Conservative decomposition: three Rz gates per controlled phase.
    # Some of these rotations are Clifford; no cancellation is credited.
    return 3 * m * (m - 1) // 2


def synthesis_cost(rotation_count, norm_budget):
    if rotation_count == 0:
        return 0, None
    epsilon = norm_budget / rotation_count
    # Leading Ross--Selinger synthesis-cost model, not a compiled upper bound.
    return math.ceil(3 * math.log2(1 / epsilon)), epsilon


def oracle_cost(M, mu, q=1):
    """Padded coherent alias sampling and unary-iteration controlled SELECT.

    L is a power of two: uniform preparation uses exact Hadamards only.
    Padding slots have zero target weight and an identity SELECT payload.
    The generic cost includes padding. It credits four T gates per Toffoli
    in the measurement-assisted convention of the cited oracle model.
    """
    ell = ceil_log2_integer(M)
    L = 1 << ell
    if not isinstance(q, int) or q < 1 or q > L or q & (q - 1):
        raise ValueError("q must be a power of two between 1 and L.")
    prep = 4 * (L // q) + 4 * (q - 1) * (mu + ell) + 4 * mu + 12 * ell
    select = 4 * L - 4
    reflection = 4 * (ell - 1)
    return {
        "padded_terms": L, "index_qubits": ell,
        "T_PREPARE": prep, "T_SELECT": select, "T_reflection": reflection,
        "T_controlled_walk": 2 * prep + select + reflection,
        "oracle_ancillas": (mu + ell) * (q - 1) + 2 * mu + 2 * ell
                            + ceil_log2_integer(L // q) + 1,
    }


def estimate_case(row, delta_energy, settings):
    M = int(row["terms_including_identity"]) - 1
    lam = float(row["lambda_nonidentity"])
    C = float(row["C"])
    if M < 2 or lam <= 0 or C < 0:
        raise ValueError("This model requires at least two nonidentity terms and lambda > 0.")
    if not 0 < delta_energy < lam:
        raise ValueError("Choose a positive target error smaller than lambda_nonidentity.")
    fraction = settings["phase_estimation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("The phase-estimation fraction must be between zero and one.")
    epsilon_circ = settings["circuit_synthesis_norm_budget"]
    if not 0 < epsilon_circ < 0.01:
        raise ValueError("Use a circuit norm budget between zero and 0.01.")
    shots = settings["shots"]
    if not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer.")
    dpe = fraction * delta_energy
    dapprox = (1 - fraction) * delta_energy
    # ||H'|| <= lambda, so its phases lie in [-pi/2, pi/2].
    dt = math.pi / (2 * lam)
    norm_pf = 2 * math.sin(dt * dapprox / 2)
    r = max(1, math.ceil(C * dt * dt / norm_pf))
    mT = max(2, ceil_log2_positive(2 * math.pi / (dt * dpe)))
    mW = max(2, ceil_log2_positive(2 * math.pi * lam / dpe))
    QT, QW = (1 << mT) - 1, (1 << mW) - 1
    # Balanced rounding of alias probabilities on a 1/(L*2**mu) grid
    # gives sum_j |p_j-p_tilde_j| <= 2**(-mu), including padding slots.
    mu = ceil_log2_positive(lam / dapprox)
    oracle = oracle_cost(M, mu, settings["qrom_tradeoff_q"])
    rotT = 2 * M * r * QT + qft_rotations(mT)
    rotW = qft_rotations(mW)
    TeT, epsT = synthesis_cost(rotT, epsilon_circ)
    TeW, epsW = synthesis_cost(rotW, epsilon_circ)
    totalT = shots * rotT * TeT
    # One initial PREPARE populates the invariant walk subspace. Nuclear
    # eigenstate preparation is a separate, excluded task for both methods.
    totalW = shots * ((QW * oracle["T_controlled_walk"])
                      + oracle["T_PREPARE"] + rotW * TeW)
    pf_bound = C * dt * dt / r
    checks = {
        "phase_energy_T_MeV": 2 * math.pi / ((1 << mT) * dt),
        "phase_energy_W_MeV": 2 * math.pi * lam / (1 << mW),
        "product_formula_spectral_error_MeV": 2 * math.asin(pf_bound / 2) / dt,
        "coefficient_error_bound_MeV": math.ldexp(lam, -mu),
        "synthesis_norm_sum_T": rotT * epsT,
        "synthesis_norm_sum_W": rotW * epsW,
    }
    tol = 1 + 2e-14
    assert checks["phase_energy_T_MeV"] <= dpe * tol
    assert checks["phase_energy_W_MeV"] <= dpe * tol
    assert checks["product_formula_spectral_error_MeV"] <= dapprox * tol
    assert checks["coefficient_error_bound_MeV"] <= dapprox * tol
    return {
        "id": row["id"], "family": row["family"], "nqubits": row["nqubits"],
        "energy_error_MeV": delta_energy, "energy_PE_MeV": dpe,
        "energy_approximation_MeV": dapprox, "identity_coefficient_MeV": row["identity_coefficient"],
        "lambda_nonidentity_MeV": lam, "C_pair_MeV2": C, "nonidentity_terms": M,
        "delta_t_MeV_inverse": dt, "r": r, "mu": mu,
        "phase_qubits_T": mT, "phase_qubits_W": mW,
        "queries_T": QT, "queries_W": QW,
        "rotation_count_T": rotT, "rotation_count_W": rotW,
        "epsilon_rotation_T": epsT, "epsilon_rotation_W": epsW,
        "T_per_rotation_T": TeT, "T_per_rotation_W": TeW,
        "T_Trotter": totalT, "T_walk": totalW, "ratio": totalT / totalW,
        "step_to_walk_gate_ratio": 2 * M * TeT / oracle["T_controlled_walk"],
        "conditional_success_lower_bound": 8 / math.pi**2 - 2 * epsilon_circ,
        "logical_qubits_T_excluding_compiler_work": row["nqubits"] + mT,
        "logical_qubits_W": row["nqubits"] + mW + oracle["oracle_ancillas"],
        "checks": checks, **oracle,
    }


# Data symbols and fit lines in Figure 6(a,b).
PAPER_MARKER_ALPHA = 0.85
PAPER_LINE_ALPHA = 0.80


def _series_colors(color, three_body=False):
    from matplotlib.colors import to_rgba
    return dict(color=to_rgba(color, PAPER_LINE_ALPHA),
                markerfacecolor=to_rgba(color, PAPER_MARKER_ALPHA),
                markeredgecolor=to_rgba("black", PAPER_MARKER_ALPHA) if three_body else "none")


PAPER_PLOT_STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
}


def plot_results(results, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    families = [("valence", "Valence space", "X", "#af4048"),
                ("NN", "NCSM (NN-only)", "s", "#3768a6"),
                ("NN3NF", "NCSM (NN+3N)", "^", "#397c50")]
    energies = sorted({r["energy_error_MeV"] for r in results}, reverse=True)
    styles = [":", "--", "-"]
    if len(energies) > len(styles):
        raise ValueError("The compact publication plot supports at most three target errors.")
    with plt.rc_context(PAPER_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.7, 3.65), layout="constrained")
        family_handles = []
        for energy, style in zip(energies, styles):
            for family, label, marker, color in families:
                rr = sorted((r for r in results if r["family"] == family
                             and r["energy_error_MeV"] == energy), key=lambda r: r["nqubits"])
                # Figure 6 uses translucent symbols and outlines the NN+3N cases.
                symbol = dict(marker=marker, ms=6.5, markeredgewidth=0.8,
                              **_series_colors(color, family == "NN3NF"))
                ax.plot([r["nqubits"] for r in rr], [r["ratio"] for r in rr],
                        ls=style, lw=1.05, **symbol)
                if energy == energies[0]:
                    family_handles.append(Line2D([], [], ls="", label=label, **symbol))
        ax.set(xscale="log", yscale="log", xlim=(3, 1100),
               xlabel="Number of system qubits $N_q$",
               ylabel=r"$R=T_{\rm QPE}^{\rm Trotter}/T_{\rm QPE}^{\rm walk}$")
        ax.set_ylim(0.5, max(r["ratio"] for r in results) * 12)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="both", ls="--", lw=0.65, color="0.83")
        family_legend = ax.legend(handles=family_handles, loc="upper left", fontsize=10,
                handlelength=1.2, handletextpad=0.45, labelspacing=0.28, borderpad=0.35)
        ax.add_artist(family_legend)
        def energy_label(e):
            return f"{e:g} MeV" if e >= 1 else f"{1000*e:g} keV"
        ax.legend(handles=[Line2D([], [], color="0.25", ls=s,
                    label=energy_label(e)) for e, s in zip(energies, styles)],
                  title=r"$\Delta E$", loc="lower right",
                  bbox_to_anchor=(0.995, 0.06), fontsize=10, title_fontsize=10,
                  handlelength=1.8, handletextpad=0.45, labelspacing=0.28, borderpad=0.35)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.03, transparent=True)
        plt.close(fig)


def figure6_axes(panel, title, *, figsize=(4.8, 3.55), label_anchors=None):
    """Use fixed margins so all four PDFs line up when included at equal width."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.154, right=0.967, bottom=0.185, top=0.965)
    label_pos = (label_anchors or {}).get(panel, (0.023, 0.965))
    ax.text(*label_pos, f"({panel}) {title}", transform=ax.transAxes,
            ha="left", va="top", fontsize=11.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=0.6))
    ax.set(xscale="log", xlim=(1, 10000), ylim=(0, 29),
           xlabel="Number of system qubits $N_q$", ylabel="T-gate count")
    ticks = range(0, 29, 4)
    ax.set_yticks(ticks, [rf"$10^{{{tick}}}$" for tick in ticks])
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, ls="--", lw=0.65, color="0.83")
    ax.set_axisbelow(True)
    return fig, ax


def figure6_legend(fig, panel, method_handles, *, anchors=None, family_anchors=None,
                   fontsize=10.5, fit_exponents=None):
    """Show the methods and applicable interaction families in each panel."""
    from matplotlib.lines import Line2D
    ax = fig.axes[0]
    styles = [("--", "none", "NN")]
    if panel in ("b", "d"):
        styles.append(("-.", "black", "NN+3N"))
    family_handles = [
        Line2D([], [], color="0.45", marker="o", ls=ls, lw=1, markersize=6,
               markeredgecolor=edge, label=label) for ls, edge, label in styles
    ]
    style = dict(bbox_transform=ax.transAxes, framealpha=0.95, facecolor="white",
                 edgecolor="none", fontsize=fontsize, handlelength=1.6, handletextpad=0.45,
                 columnspacing=1, labelspacing=0.32, borderpad=0.25, borderaxespad=0)
    family_anchor = (family_anchors or {}).get(panel, (0.02, 0.845 if panel in ("a", "b") else 0.745))
    family_key = ax.legend(handles=family_handles, loc="upper left", bbox_to_anchor=family_anchor,
                           ncol=len(family_handles), **style)
    ax.add_artist(family_key)
    anchor = (anchors or {}).get(panel, (0.978, 0.038))
    if fit_exponents is not None:
        from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
        from matplotlib.text import Text

        # All offsets are in type points, so the key scales with the figure.
        families = list(fit_exponents)
        step = 1.17 * fontsize
        width = (10.8 if len(families) == 1 else 15.2) * fontsize
        key = DrawingArea(width, (len(method_handles) + 1) * step, 0, 0)
        top = (len(method_handles) + 0.5) * step
        def label(x, y, value, align="left"):
            key.add_artist(Text(x, y, value, ha=align, va="center", fontsize=fontsize))

        label(0, top, r"$T \propto N_q^{\alpha}$")
        centers = [9.7 * fontsize, 13.3 * fontsize]
        for col, family in enumerate(families):
            label(centers[col], top, "NN+3N" if family == "NN3NF" else "NN", "center")
        for i, handle in enumerate(method_handles):
            y = top - (i + 1) * step
            key.add_artist(Line2D([0.45 * fontsize], [y], marker=handle.get_marker(),
                                  color=handle.get_color(), markeredgecolor="none", ls=""))
            label(1.3 * fontsize, y, handle.get_label())
            for x, family in zip(centers, families):
                value = fit_exponents[family][i]
                label(x, y, "–" if value is None else f"{value:.1f}", "center")
        box = AnchoredOffsetbox(loc="lower right", child=key, pad=0.15, borderpad=0,
                                bbox_to_anchor=anchor, bbox_transform=ax.transAxes, frameon=True)
        box.patch.set(facecolor="white", alpha=0.88, edgecolor="none")
        ax.add_artist(box)
        return
    ax.legend(handles=method_handles, loc="lower right", bbox_to_anchor=anchor,
              ncol=2 if panel in ("a", "b") else 1, **style)


def plot_absolute_counts(results, output_dir, energy_error, *, figsize=(4.8, 3.55),
                         label_anchors=None, legend_anchors=None, family_anchors=None,
                         legend_fontsize=10.5, show=False):
    """Fig. 6(c,d): evaluated costs at one common absolute energy error."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = [r for r in results if r["energy_error_MeV"] == energy_error]
    if not selected:
        raise ValueError("figure6_energy_MeV must be in energy_errors_MeV.")
    value, unit = (energy_error, "MeV") if energy_error >= 1 else (1000 * energy_error, "keV")
    energy_label = rf"$\Delta E={value:g}$ {unit}"
    methods = [("T_Trotter", "Trotter (pair bound)", "s", "#af4048"),
               ("T_walk", "Qubitization", "o", "#3768a6")]
    outputs = []
    with plt.rc_context(PAPER_PLOT_STYLE):
        for valence in (True, False):
            panel = "c" if valence else "d"
            space = "Valence" if valence else "No-core"
            fig, ax = figure6_axes(panel, f"{space}, QPE\n{energy_label}", figsize=figsize,
                                   label_anchors=label_anchors)
            families = ["valence"] if valence else ["NN", "NN3NF"]
            for family in families:
                rows = sorted((r for r in selected if r["family"] == family),
                              key=lambda r: r["nqubits"])
                if not rows:
                    continue
                three_body = family == "NN3NF"
                for key, label, marker, color in methods:
                    # Connect evaluated cases only; do not extrapolate a fit.
                    ax.plot([r["nqubits"] for r in rows],
                            [math.log10(r[key]) for r in rows],
                            marker=marker, ms=6, markeredgewidth=0.8,
                            linestyle="-." if three_body else "--", lw=1,
                            **_series_colors(color, three_body))
            handles = [Line2D([], [], marker=marker, ls="", label=label,
                              **_series_colors(color))
                       for _, label, marker, color in methods]
            figure6_legend(fig, panel, handles, anchors=legend_anchors,
                           family_anchors=family_anchors, fontsize=legend_fontsize)
            filename = ("Tgate_counts_valence_TrotterQubitization.pdf" if valence
                        else "Tgate_counts_nocore_TrotterQubitization.pdf")
            path = Path(output_dir) / filename
            fig.savefig(path, transparent=True)
            if show:
                plt.show()
            plt.close(fig)
            outputs.append(path)
    return outputs


def run(results_path=None, output_dir=None, settings=None, paper_figures=None, figure6_layout=None):
    repo = Path(__file__).resolve().parents[1]
    results_path = Path(results_path) if results_path else repo / "data" / "operator_summaries.json"
    output_dir = Path(output_dir) if output_dir else repo / "build"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(SETTINGS if settings is None else settings)
    figure_energy = settings.setdefault("figure6_energy_MeV", 0.1)
    if figure_energy not in settings["energy_errors_MeV"]:
        raise ValueError("Include figure6_energy_MeV in energy_errors_MeV.")
    payload = json.loads(results_path.read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    qsvt = normalization_report(rows)
    rows = [r for r in rows if r.get("main_figure", True)]
    for row in rows:
        if not {"identity_coefficient", "lambda_nonidentity"} <= row.keys():
            raise ValueError(f"Missing verified identity-shift data: {row['id']}")
    results = [estimate_case(row, energy, settings)
               for energy in settings["energy_errors_MeV"] for row in rows]
    report = {
        "model": "identity-shifted QPE, matched absolute energy error, padded q=1 alias oracle",
        "settings": settings,
        "figure6": {
            "panels": ["c", "d"], "energy_error_MeV": figure_energy,
            "case_ids": [r["id"] for r in rows],
            "curves": ["T_Trotter", "T_walk"],
            "connections": "evaluated cases only; no extrapolation",
        },
        "source_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "cost_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assumptions": [
            "Accuracy is relative to the retained coefficient-screened benchmark Hamiltonian; model-space and interaction uncertainties are excluded.",
            "Per spectral sample conditional on an encoded eigenstate; equal usable target overlap is assumed, not established.",
            "PE and Hamiltonian approximation receive additive fractions of the same absolute energy error.",
            "Identical Trotter base unitary is repeated for every QPE power; no independently approximated long-time powers.",
            "Coefficient rounding preserves normalization and gives l1 error at most 2**(-mu).",
            "Power-of-two padding makes uniform alias preparation exact; zero-weight SELECT payloads are identity.",
            "All synthesized rotations share a per-circuit operator-norm budget; QFT rotations are included.",
            "The leading logarithmic synthesis formula is a cost estimate, not a gate-compiled certificate.",
            "Generic unary SELECT counts arbitrary controlled Pauli strings as Clifford, including NN+3N strings.",
            "Four-T measurement-assisted Toffoli cost convention; Clifford cost and physical error correction are excluded.",
            "Nuclear eigenstate preparation and changes in ground-state overlap under approximation are excluded.",
            "C_pair supplies sufficient first-order step counts; the ratio does not establish an optimal-algorithm crossover.",
        ],
        "references": ["https://arxiv.org/abs/1805.03662", "https://arxiv.org/abs/2511.21867"],
        "results": results,
        "qsvt_normalizations": qsvt,
    }
    (output_dir / "qpe_cost_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    table = output_dir / "table_qsvt.tex"
    table.write_text(normalization_table_tex(qsvt))
    pdf = output_dir / "qpe_Tcount_ratio.pdf"
    plot_results(results, pdf)
    outputs = [pdf] + plot_absolute_counts(results, output_dir, figure_energy,
                                           **(figure6_layout or {}))
    if paper_figures is not None:
        import shutil
        for output in outputs:
            shutil.copy2(output, Path(paper_figures) / output.name)
        shutil.copy2(table, Path(paper_figures).parent / table.name)
    for energy in settings["energy_errors_MeV"]:
        selected = [r for r in results if r["energy_error_MeV"] == energy]
        print(f"Delta E={energy:g} MeV: {len(selected)} cases, "
              f"R={min(r['ratio'] for r in selected):.3g}--{max(r['ratio'] for r in selected):.3g}; "
              f"mu={min(r['mu'] for r in selected)}--{max(r['mu'] for r in selected)}")
    return report


if __name__ == "__main__":
    run()
