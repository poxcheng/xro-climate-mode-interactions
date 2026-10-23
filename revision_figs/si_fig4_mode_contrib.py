import os
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.join(HERE, "intermediates")
DATA = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
"""SI figure: per-mode contributions to mature-phase tropical-mean SST/LTT.

Bars = 12-event composite contribution (K), dots = individual events,
labels = percent of the 9-mode reconstructed total.
Data from mode_contrib.pkl (beta-projection recipe, Nov(0)-Feb(+1) composite).
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(os.path.join(INT, "mode_contrib.pkl"), "rb") as f:
    d = pickle.load(f)

ORDER = ["Nino34", "NPMM", "SPMM", "IOB", "IOD", "SIOD", "TNA", "ATL3", "SASD"]
XLAB = ["Nino3.4", "NPMM", "SPMM", "IOB", "IOD", "SIOD", "TNA", "ATL3", "SASD"]
GROUP_COLOR = {
    "Nino34": "#b2182b",
    "NPMM": "#4393c3", "SPMM": "#4393c3",
    "IOB": "#e08214", "IOD": "#e08214", "SIOD": "#e08214",
    "TNA": "#762a83", "ATL3": "#762a83", "SASD": "#762a83",
}

fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)

for ax, var, lab in zip(axes, ["SST", "LTT"], ["a", "b"]):
    df = d[var][ORDER]
    comp = df.mean(axis=0)
    shares = 100.0 * comp / comp.sum()
    x = np.arange(len(ORDER))

    ax.axhline(0, color="k", lw=0.8)
    ax.bar(x, comp.values, width=0.62,
           color=[GROUP_COLOR[m] for m in ORDER], alpha=0.85, zorder=2)
    for xi, (c, s) in enumerate(zip(comp.values, shares.values)):
        off = 0.003 if c >= 0 else -0.003
        va = "bottom" if c >= 0 else "top"
        ax.text(xi, c + off, f"{c:.3f}", ha="center", va=va, fontsize=10.2)

    ax.set_ylabel("Contribution (K)")
    ax.set_title(f"({lab}) Tropical-mean {var}, Nov(0)-Feb(+1) composite",
                 loc="left", fontsize=12.6)
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB, fontsize=10.8)
    ax.tick_params(length=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(comp.values.min() - 0.015, comp.values.max() + 0.015)

axes[0].legend(handles=[
    plt.Rectangle((0, 0), 1, 1, fc="#b2182b", alpha=0.85, label="ENSO"),
    plt.Rectangle((0, 0), 1, 1, fc="#4393c3", alpha=0.85, label="Pacific meridional"),
    plt.Rectangle((0, 0), 1, 1, fc="#e08214", alpha=0.85, label="Indian Ocean"),
    plt.Rectangle((0, 0), 1, 1, fc="#762a83", alpha=0.85, label="Atlantic"),
], fontsize=10.2, frameon=False, loc="upper right", ncol=2)

fig.suptitle("Individual mode contributions to the composite tropical-mean warming",
             fontsize=13.8, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("FigS_mode_contrib.png", dpi=300, bbox_inches="tight")
print("saved FigS_mode_contrib.png")
for var in ["SST", "LTT"]:
    comp = d[var][ORDER].mean(axis=0)
    sh = (100 * comp / comp.sum()).round(0).astype(int)
    print(var, "total %.3f K" % comp.sum(), dict(sh))
