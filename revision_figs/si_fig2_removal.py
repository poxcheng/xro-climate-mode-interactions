import os
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.join(HERE, "intermediates")
DATA = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
"""Replot SX (Supplementary Fig. 2) from rm1d.pkl with a title that names both
the observed operator and the CMIP operators. Layout follows rm1d_final.png:
2x3, row-shared y, 3-monthly rotated ticks, figure-level bottom legend."""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

d = pickle.load(open(os.path.join(INT, "rm1d.pkl"), "rb"))
obs, raw, corr, nM = d["obs"], d["raw"], d["corr"], d["n"]

BLOCKS = ["C1", "C2", "L8off"]
OTHERS = {"C1": ["C2", "L8", "LENSO"],
          "C2": ["C1", "L8", "LENSO"],
          "L8off": ["C1", "C2", "LENSO"]}
CC = {"C1": "tab:green", "C2": "tab:blue", "L8": "tab:purple", "LENSO": "tab:orange"}

rel = np.arange(24)
xt = np.arange(0, 24, 3)
xl = ["Jan(0)", "Apr(0)", "Jul(0)", "Oct(0)",
      "Jan(+1)", "Apr(+1)", "Jul(+1)", "Oct(+1)"]

fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=True, sharey="row")
for ri, v in enumerate(("SST", "LTT")):
    for ci, B in enumerate(BLOCKS):
        ax = axes[ri, ci]
        R = np.stack(raw[(v, B)])
        ax.fill_between(rel, np.percentile(R, 10, axis=0), np.percentile(R, 90, axis=0),
                        color="tab:red", alpha=0.15, lw=0)
        ax.plot(rel, R.mean(axis=0), color="tab:red", lw=2)
        for O in OTHERS[B]:
            Cv = np.stack(corr[(v, B, O)])
            ax.plot(rel, Cv.mean(axis=0), color=CC[O], lw=1.4, ls="--")
        ax.plot(rel, obs[(v, B)], color="k", lw=2.2)
        ax.axhline(0, color="0.6", lw=0.7)
        lab = {"C1": r"$C_1$=0", "C2": r"$C_2$=0", "L8off": r"$L_8^{off}$=0"}[B]
        ax.set_title(f"({'abcdef'[ri*3+ci]}) {v}: {lab} $-$ BASE", loc="left", fontsize=10.5)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=8.5, rotation=30)
        if ci == 0:
            ax.set_ylabel(f"tropical-mean {v} diff (K)")

handles = [
    Line2D([], [], color="k", lw=2.2, label="OBS operator"),
    Line2D([], [], color="tab:red", lw=2, label="CMIP mean (raw operator)"),
    Patch(facecolor="tab:red", alpha=0.15, label="CMIP 10-90%"),
    Line2D([], [], color=CC["C1"], lw=1.4, ls="--", label=r"removal after $C_1\to$obs"),
    Line2D([], [], color=CC["C2"], lw=1.4, ls="--", label=r"removal after $C_2\to$obs"),
    Line2D([], [], color=CC["L8"], lw=1.4, ls="--", label=r"removal after $L_8\to$obs"),
    Line2D([], [], color=CC["LENSO"], lw=1.4, ls="--", label=r"removal after $L_{ENSO}\to$obs"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Coupling-removal experiments in the observed operator and {nM} CMIP operators\n"
             "tropical-mean composites of 9 ONI El Niño events, 1979-2010",
             fontsize=12)
fig.tight_layout(rect=[0, 0.06, 1, 0.94])
fig.savefig("rm1d_final.png", dpi=160, bbox_inches="tight")
print("saved rm1d_final.png")
