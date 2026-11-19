import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def plot(results, settings, output):
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.labelsize": 12, "legend.fontsize": 10})
    fig, axes = plt.subplots(len(results), 2, figsize=(9.2, 2.35 * len(results)),
                             squeeze=False, layout="constrained")
    colors = {1: "#3768a6", 2: "#b6581b", 4: "#397c50"}
    markers = {1: "o", 2: "s", 4: "^"}
    for row, data in zip(axes, results):
        for order in settings["orders"]:
            values = [v for v in data["results"] if v["order"] == order]
            error = [v["operator_norm_error"] for v in values]
            for ax, xkey in zip(row, ["steps", "pauli_rotations"]):
                ax.loglog([v[xkey] for v in values], error, color=to_rgba(colors[order], 0.80),
                          marker=markers[order], markerfacecolor=to_rgba(colors[order], 0.85),
                          markeredgecolor="none", ms=4, lw=1.3, label=f"Order {order}")
            fit_x = np.array(settings["fit_steps"], dtype=float)
            fitted = data["fits"][str(order)]
            row[0].loglog(fit_x, fitted["prefactor"] * fit_x ** fitted["power"],
                         color="black", lw=0.8, ls="--", zorder=4)
        for ax in row:
            ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
            ax.set_ylabel(r"$\|V_p(t,r)-e^{-itH'}\|_2$")
            ax.grid(True, which="major", color="0.88", lw=0.5)
            ax.set_ylim(1e-9, 1)
        row[0].set_xlabel(r"Trotter steps $r$")
        row[1].set_xlabel("Pauli rotations")
        fit_steps = settings["fit_steps"]
        row[0].set_title(rf"(a) Fits to $r^s$ over $r={min(fit_steps)}$–${max(fit_steps)}$",
                         loc="left", fontsize=11)
        row[1].set_title("(b) Error versus rotation count", loc="left", fontsize=11)
        fit_labels = [rf"$p={p},\ s={data['fits'][str(p)]['power']:.2f}$"
                      for p in settings["orders"]]
        handles, _ = row[0].get_legend_handles_labels()
        row[0].legend(handles, fit_labels,
                      loc="lower left", frameon=False, fontsize=10,
                      handlelength=1.45, handletextpad=0.45, labelspacing=0.12,
                      borderpad=0.15, borderaxespad=0.4, alignment="left")
        row[1].text(0.025, 0.065, data["plot_label"] + "\n" +
                    rf"$t={data['time_MeV_inverse']:g}\,\mathrm{{MeV}}^{{-1}}$; $M={data['nonidentity_terms']}$",
                    transform=row[1].transAxes, ha="left", va="bottom", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Title": "6He product-formula convergence", "Creator": "NuQuLib benchmark"})
    plt.close(fig)
