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


# Data symbols and fit lines of the fixed-parameter panels (Fig. 7 of the revised
# manuscript; Fig. 6(a,b) of the earlier four-panel figure).
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
                # Figs. 7 and 8 use translucent symbols and outline the NN+3N cases.
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


def figure6_axes(panel, title, *, figsize=(4.8, 3.55), label_anchors=None, label=None,
                 ylim=(0, 29), yticks=range(0, 29, 4), ylabel="T-gate count",
                 title_outside=False, right_margin=None):
    """Use fixed margins so all panel PDFs line up when included at equal width.

    ``panel`` selects the layout anchors; ``label`` is the printed panel letter
    and defaults to ``panel``. The vertical axis holds log10 values, so
    ``ylim`` and ``yticks`` are exponents of ten. Margins are fixed in inches,
    so smaller panels keep the same room for tick and axis labels; at the
    default size they equal the original fractional margins. With
    ``title_outside`` the panel label is placed above the axes, which leaves
    the plotting area to the data and legends in small panels.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    width, height = figsize
    # Inch margins equal the original fractions of the 4.8 x 3.55 in panels,
    # so the default size reproduces the earlier figures exactly.
    top_margin = 0.32 if title_outside else 0.035 * 3.55
    # ``right_margin`` (inches) narrows the gap to a neighboring panel; it
    # must leave room for half of the last x tick label.
    right = 0.033 * 4.8 if right_margin is None else right_margin
    fig.subplots_adjust(left=0.154 * 4.8 / width, right=1 - right / width,
                        bottom=0.185 * 3.55 / height, top=1 - top_margin / height)
    text = f"({label or panel}) {title}"
    if title_outside:
        ax.set_title(text, loc="left", fontsize=11.5, pad=5)
    else:
        label_pos = (label_anchors or {}).get(panel, (0.023, 0.965))
        ax.text(*label_pos, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=11.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=0.6))
    ax.set(xscale="log", xlim=(1, 10000), ylim=ylim,
           xlabel="Number of system qubits $N_q$", ylabel=ylabel)
    ax.set_yticks(list(yticks), [rf"$10^{{{tick}}}$" for tick in yticks])
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, ls="--", lw=0.65, color="0.83")
    ax.set_axisbelow(True)
    return fig, ax


def figure6_legend(fig, panel, method_handles, *, anchors=None, family_anchors=None,
                   fontsize=10.5, fit_exponents=None, three_body=None, family_ncol=None):
    """Show the methods and applicable interaction families in each panel.

    ``three_body`` adds the NN+3N key; by default it follows the no-core
    panels (b) and (d) of the combined four-panel layout. ``family_ncol``
    sets the columns of the interaction key (default: one row).
    """
    from matplotlib.lines import Line2D
    ax = fig.axes[0]
    styles = [("--", "none", "NN")]
    if three_body is None:
        three_body = panel in ("b", "d")
    if three_body:
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
                           ncol=family_ncol or len(family_handles), **style)
    ax.add_artist(family_key)
    anchor = (anchors or {}).get(panel, (0.978, 0.038))
    if fit_exponents is not None:
        from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
        from matplotlib.text import Text

        # All offsets are in type points, so the key scales with the figure.
        families = list(fit_exponents)
        step = 1.27 * fontsize
        width = (11.4 if len(families) == 1 else 15.9) * fontsize
        key = DrawingArea(width, (len(method_handles) + 1) * step, 0, 0)
        top = (len(method_handles) + 0.5) * step
        def label(x, y, value, align="left"):
            key.add_artist(Text(x, y, value, ha=align, va="center", fontsize=fontsize))

        label(0, top, r"$T \propto N_q^{\alpha}$")
        centers = [9.8 * fontsize, 13.7 * fontsize]
        for col, family in enumerate(families):
            header = r"$\alpha_{\mathrm{NN+3N}}$" if family == "NN3NF" else r"$\alpha_{\mathrm{NN}}$"
            label(centers[col], top, header, "center")
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


# Vertical range of the fixed-accuracy panels, in powers of ten. The largest
# Trotter estimates exceed 1e28, so the range extends above the
# fixed-parameter panels to keep every evaluated point inside the axes.
FIXED_ACCURACY_YLIM = (5, 30.5)
FIXED_ACCURACY_YTICKS = range(8, 30, 4)

# File names of the split layout: one figure for the fixed-accuracy QPE
# comparison, with panels (a) valence, (b) no-core, and (c) cost ratios.
SPLIT_FILENAMES = {
    "valence": "QPE_fixed_accuracy_valence.pdf",
    "nocore": "QPE_fixed_accuracy_nocore.pdf",
    "ratio": "QPE_fixed_accuracy_ratio.pdf",
}

# The split figure places its three panels side by side, each printed at
# about 0.32 of the text width. The smaller canvas keeps the printed text
# near 7 pt. Panel titles sit above the axes, and the interaction key is a
# single column at the upper left, where the data leave room. Anchors are
# fractions of the axes, keyed by layout panel.
SPLIT_LAYOUT = {
    "figsize": (3.2, 3.1),
    "legend_fontsize": 9.5,
    "title_outside": True,
    "family_ncol": 1,
    "label_anchors": {"c": (0.03, 0.965), "d": (0.03, 0.965)},
    # Method key: lower-right corner. The valence data end at about 60% of
    # the height, so its key sits above them; the no-core key stays low.
    "legend_anchors": {"c": (0.975, 0.70), "d": (0.975, 0.035)},
    "family_anchors": {"c": (0.025, 0.975), "d": (0.025, 0.975)},
    "ratio_ylim": (0, 18),
    "right_margin": 0.12,
}


def plot_absolute_counts(results, output_dir, energy_error, *, figsize=(4.8, 3.55),
                         label_anchors=None, legend_anchors=None, family_anchors=None,
                         legend_fontsize=10.5, show=False, display_labels=None,
                         filenames=None, ylim=FIXED_ACCURACY_YLIM,
                         yticks=FIXED_ACCURACY_YTICKS, title_outside=False,
                         family_ncol=None, right_margin=None):
    """Evaluated QPE costs at one common absolute energy error.

    By default the panels are (c) and (d) of the combined four-panel figure.
    ``display_labels`` maps the layout keys "c" and "d" to printed letters,
    and ``filenames`` maps "valence" and "nocore" to output names; the split
    layout prints (a) and (b) while reusing the (c,d) anchors.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = [r for r in results if r["energy_error_MeV"] == energy_error]
    if not selected:
        raise ValueError("figure6_energy_MeV must be in energy_errors_MeV.")
    value, unit = (energy_error, "MeV") if energy_error >= 1 else (1000 * energy_error, "keV")
    energy_label = rf"$\Delta E={value:g}$ {unit}"
    # Compact panels use short method names; their caption states that the
    # Trotter step counts come from the pairwise commutator bound.
    trotter_label = "Trotter" if title_outside else "Trotter (pair bound)"
    methods = [("T_Trotter", trotter_label, "s", "#af4048"),
               ("T_walk", "Qubitization", "o", "#3768a6")]
    outputs = []
    with plt.rc_context(PAPER_PLOT_STYLE):
        for valence in (True, False):
            panel = "c" if valence else "d"
            space = "Valence" if valence else "No-core"
            # A title above the axes has room for one line only.
            title = (f"{space}, {energy_label}" if title_outside
                     else f"{space}, QPE\n{energy_label}")
            fig, ax = figure6_axes(panel, title, figsize=figsize,
                                   label_anchors=label_anchors,
                                   label=(display_labels or {}).get(panel),
                                   ylim=ylim, yticks=yticks, title_outside=title_outside,
                                   right_margin=right_margin)
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
                           family_anchors=family_anchors, fontsize=legend_fontsize,
                           three_body=not valence, family_ncol=family_ncol)
            names = {"valence": "Tgate_counts_valence_TrotterQubitization.pdf",
                     "nocore": "Tgate_counts_nocore_TrotterQubitization.pdf",
                     **(filenames or {})}
            path = Path(output_dir) / names["valence" if valence else "nocore"]
            fig.savefig(path, transparent=True)
            if show:
                plt.show()
            plt.close(fig)
            outputs.append(path)
    return outputs


def plot_ratio_panel(results, output, *, figsize=(4.8, 3.55), label="c", label_anchor=None,
                     legend_fontsize=10.5, ylim=(0, 16), yticks=range(0, 15, 2), show=False,
                     title_outside=False, right_margin=None):
    """Trotter-to-qubitization ratios drawn in the frame of the T-count panels.

    The axes margins and horizontal range match ``figure6_axes`` so that this
    panel lines up with the absolute counts of the split figure. Colors and
    markers distinguish interaction families; line styles distinguish the
    target energy errors.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # A small panel with the title above the axes uses short family names;
    # the caption states that the NN and NN+3N series are no-core spaces.
    no_core = "" if title_outside else "No-core "
    families = [("valence", "Valence", "X", "#af4048"),
                ("NN", f"{no_core}NN", "s", "#3768a6"),
                ("NN3NF", f"{no_core}NN+3N", "^", "#397c50")]
    energies = sorted({r["energy_error_MeV"] for r in results}, reverse=True)
    styles = [":", "--", "-"]
    if len(energies) > len(styles):
        raise ValueError("The ratio panel supports at most three target errors.")
    # Keep the upper part of the axes free for the panel label and the keys.
    top = max(math.log10(r["ratio"]) for r in results)
    if top > (0.62 if title_outside else 0.65) * ylim[1]:
        raise ValueError(f"Raise ylim: the largest ratio is 10^{top:.2f}.")
    with plt.rc_context(PAPER_PLOT_STYLE):
        fig, ax = figure6_axes("c", "Trotter/qubitization", figsize=figsize,
                               label_anchors={"c": label_anchor or (0.023, 0.965)},
                               label=label, ylim=ylim, yticks=yticks,
                               ylabel=r"$T_{\rm QPE}^{\rm Trotter}/T_{\rm QPE}^{\rm walk}$",
                               title_outside=title_outside, right_margin=right_margin)
        family_handles = []
        for energy, style in zip(energies, styles):
            for family, name, marker, color in families:
                rows = sorted((r for r in results if r["family"] == family
                               and r["energy_error_MeV"] == energy), key=lambda r: r["nqubits"])
                symbol = dict(marker=marker, ms=6, markeredgewidth=0.8,
                              **_series_colors(color, family == "NN3NF"))
                # Connect evaluated cases only; do not extrapolate a fit.
                ax.plot([r["nqubits"] for r in rows], [math.log10(r["ratio"]) for r in rows],
                        ls=style, lw=1, **symbol)
                if energy == energies[0]:
                    family_handles.append(Line2D([], [], ls="", label=name, **symbol))
        key_style = dict(framealpha=0.95, facecolor="white", edgecolor="none",
                         fontsize=legend_fontsize, handlelength=1.6, handletextpad=0.45,
                         labelspacing=0.28, borderpad=0.3, borderaxespad=0.3)
        # With the title above the axes, both keys share the empty band
        # above the data; otherwise the energy key sits at the lower right.
        family_key = ax.legend(handles=family_handles, loc="upper left",
                               bbox_to_anchor=(0.0, 1.0 if title_outside else 0.88),
                               **key_style)
        ax.add_artist(family_key)
        def energy_label(e):
            return f"{e:g} MeV" if e >= 1 else f"{1000*e:g} keV"
        ax.legend(handles=[Line2D([], [], color="0.25", ls=s, label=energy_label(e))
                           for e, s in zip(energies, styles)],
                  title=r"$\Delta E$", title_fontsize=legend_fontsize,
                  loc="upper right" if title_outside else "lower right", **key_style)
        fig.savefig(output, transparent=True)
        if show:
            plt.show()
        plt.close(fig)
    return Path(output)


def run(results_path=None, output_dir=None, settings=None, paper_figures=None, figure6_layout=None,
        split_layout=True):
    """Write the cost report, the QSVT table, and the QPE figures.

    The combined layout gives panels (c,d) of the earlier four-panel figure
    and a separate ratio figure. With ``split_layout`` the same results are
    also drawn as Fig. 8 of the revised manuscript: (a) valence and (b)
    no-core T-gate counts at ``figure6_energy_MeV`` and (c) the ratios at all
    target errors. The panels use ``SPLIT_LAYOUT`` for three panels side by
    side; pass a dictionary with the same keys as ``split_layout`` to change
    it. Names such as ``figure6_energy_MeV`` keep the earlier numbering so
    that existing settings and reports remain valid.
    """
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
    layout = dict(figure6_layout or {})
    outputs = [pdf] + plot_absolute_counts(results, output_dir, figure_energy, **layout)
    if split_layout:
        split = dict(SPLIT_LAYOUT if split_layout is True else split_layout)
        ratio_ylim = split.pop("ratio_ylim", (0, 16))
        outputs += plot_absolute_counts(results, output_dir, figure_energy, **split,
                                        display_labels={"c": "a", "d": "b"},
                                        filenames=SPLIT_FILENAMES)
        outputs.append(plot_ratio_panel(
            results, output_dir / SPLIT_FILENAMES["ratio"],
            figsize=split["figsize"],
            label_anchor=split["label_anchors"].get("c"),
            legend_fontsize=split["legend_fontsize"],
            title_outside=split.get("title_outside", False),
            right_margin=split.get("right_margin"),
            ylim=ratio_ylim))
        report["split_figure"] = {
            "panels": {"a": SPLIT_FILENAMES["valence"], "b": SPLIT_FILENAMES["nocore"],
                       "c": SPLIT_FILENAMES["ratio"]},
            "energy_error_MeV_ab": figure_energy,
            "energy_errors_MeV_c": settings["energy_errors_MeV"],
            "log10_ylim_ab": list(FIXED_ACCURACY_YLIM),
        }
        (output_dir / "qpe_cost_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
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
