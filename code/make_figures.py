#!/usr/bin/env python3
"""Figure generation for the PMB Topical Review:
'Biophysics of Tumor Treating Fields: capacitive coupling and dielectric
 mechanisms of a field-based cancer therapy.'

All quantitative figures are computed from standard, textbook biophysical
models (single-shell dielectric cell model; Schwan transmembrane-potential
equation; Maxwell/Wagner interfacial polarisation; Clausius-Mossotti /
dielectrophoresis theory). No experimental or patient data are used; the
figures are illustrative reproductions of established electro-physics and are
fully reproducible from this script.

References for the models used:
- Schwan H P 1957 (induced transmembrane potential)
- Foster K R, Schwan H P 1989 (dielectric properties, beta-dispersion)
- Jones T B 1995 Electromechanics of Particles (Clausius-Mossotti, DEP)
- Gimsa J, Wachner D 1998/2001 (single-shell field/torque)
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge
from matplotlib.collections import LineCollection

mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "font.family": "serif",
    "mathtext.fontset": "cm", "figure.dpi": 200,
})

EPS0 = 8.8541878128e-12          # F/m
TT_LO, TT_HI = 100e3, 400e3      # TTFields therapeutic band (Hz)
COL_BAND = "#f2c9c0"
COL_A, COL_B, COL_C = "#1f5c99", "#c0392b", "#2e8b57"


def eps_star(eps_r, sigma, w):
    """Complex permittivity eps' - j sigma/omega (rad convention)."""
    return eps_r * EPS0 - 1j * sigma / w


def shelled_sphere_eps(w, R, d, eps_mem, sig_mem, eps_cyt, sig_cyt):
    """Effective complex permittivity of a single-shell (membrane+cytoplasm)
    sphere - standard Maxwell-Wagner single-shell result (Jones 1995)."""
    gamma = R / (R - d)
    em = eps_star(eps_mem, sig_mem, w)
    ec = eps_star(eps_cyt, sig_cyt, w)
    f = (ec - em) / (ec + 2 * em)
    return em * (gamma**3 + 2 * f) / (gamma**3 - f)


def cm_factor(w, eps_p, eps_med, sig_med):
    """Clausius-Mossotti factor of the equivalent sphere in the medium."""
    ep = eps_p
    emed = eps_star(eps_med, sig_med, w)
    return (ep - emed) / (ep + 2 * emed)


def core_field_ratio(w, R, d, eps_mem, sig_mem, eps_cyt, sig_cyt,
                     eps_med, sig_med):
    """|E_cyt / E_ext| inside the cytoplasm core of a concentric single-shell
    sphere (membrane shell + cytoplasm core) in a uniform AC field. Exact
    Laplace solution for coated dielectric sphere (Kotnik & Miklavcic 2000;
    Jones 1995). e3=medium, e1=membrane shell, e2=cytoplasm core."""
    e1 = eps_star(eps_mem, sig_mem, w)   # membrane
    e2 = eps_star(eps_cyt, sig_cyt, w)   # cytoplasm core
    e3 = eps_star(eps_med, sig_med, w)   # medium
    rho = ((R - d) / R) ** 3
    D = (e1 + 2 * e3) * (e2 + 2 * e1) + 2 * rho * (e3 - e1) * (e1 - e2)
    return np.abs(9 * e3 * e1 / D)


def transmembrane_potential(w, R, Cm, sig_cyt, sig_med, E0=100.0):
    """Schwan induced transmembrane potential at the pole (cos th = 1).
    dVm = 1.5 R E / sqrt(1 + (w tau)^2), tau = R Cm (rho_i + rho_e/2).
    E0 in V/m (100 V/m = 1 V/cm)."""
    rho_i, rho_e = 1.0 / sig_cyt, 1.0 / sig_med
    tau = R * Cm * (rho_i + rho_e / 2.0)
    return 1.5 * R * E0 / np.sqrt(1 + (w * tau) ** 2), tau


# --- baseline cancer-cell-like parameters -----------------------------------
d       = 5e-9          # membrane thickness (m)
eps_mem = 5.7           # -> Cm ~ 1 uF/cm^2
sig_mem = 1e-7          # S/m (near-insulating membrane)
eps_cyt = 60.0
sig_cyt = 0.3           # S/m
eps_med = 80.0
sig_med = 0.5           # S/m (extracellular / ECM)
Cm      = eps_mem * EPS0 / d      # ~1.0e-2 F/m^2

freq = np.logspace(3, 8, 1200)   # 1 kHz - 100 MHz
w = 2 * np.pi * freq


def shade_band(ax):
    ax.axvspan(TT_LO, TT_HI, color=COL_BAND, alpha=0.75, lw=0, zorder=0)
    ax.text(np.sqrt(TT_LO * TT_HI), ax.get_ylim()[1], "TTFields\nband",
            ha="center", va="top", fontsize=7, color="#7a2016")


# ============================================================================
# FIGURE 1 - concept schematic
# ============================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    # left: transducer arrays + tissue with cell
    ax.add_patch(Rectangle((0.2, 0.5), 3.2, 3.0, fc="#fbeee6", ec="#b9967a"))
    ax.add_patch(Rectangle((0.2, 3.3), 3.2, 0.25, fc="#8a8f99", ec="k"))
    ax.add_patch(Rectangle((0.2, 0.45), 3.2, 0.25, fc="#8a8f99", ec="k"))
    ax.text(1.8, 3.75, "transducer array", ha="center", fontsize=7)
    ax.text(1.8, 0.2, "transducer array", ha="center", fontsize=7)
    for x in np.linspace(0.6, 3.0, 7):
        ax.annotate("", xy=(x, 0.75), xytext=(x, 3.25),
                    arrowprops=dict(arrowstyle="-", color=COL_A, lw=0.6, alpha=0.5))
    ax.add_patch(Circle((1.8, 2.0), 0.45, fc="#d9e6f2", ec=COL_A, lw=1.3))
    ax.add_patch(Circle((1.8, 2.0), 0.40, fc="#eef4fa", ec="none"))
    ax.text(1.8, 1.25, "cancer cell", ha="center", fontsize=7)
    ax.text(1.8, 3.05, r"$\sim$1--3 V/cm, 100--400 kHz", ha="center",
            fontsize=6.5, color=COL_A)

    # middle arrow
    ax.annotate("", xy=(4.3, 2.0), xytext=(3.6, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))

    # centre: single-shell dielectric model
    cx, cy = 6.0, 2.0
    ax.add_patch(Circle((cx, cy), 1.15, fc="#fbeee6", ec=COL_B, lw=2.2))
    ax.add_patch(Circle((cx, cy), 1.05, fc="#eef4fa", ec="none"))
    ax.text(cx, cy + 0.35, "cytoplasm", ha="center", fontsize=7)
    ax.text(cx, cy + 0.05, r"$\varepsilon_{\rm cyt},\,\sigma_{\rm cyt}$",
            ha="center", fontsize=7)
    ax.text(cx, cy - 1.45, "capacitive\nmembrane", ha="center", fontsize=6.8,
            color=COL_B)
    ax.annotate("", xy=(cx + 1.15, cy + 0.75), xytext=(cx + 1.9, cy + 1.2),
                arrowprops=dict(arrowstyle="-", color=COL_B, lw=0.7))
    ax.text(cx + 1.95, cy + 1.25, r"$C_m\!\approx\!1\,\mu$F/cm$^2$",
            fontsize=6.5, color=COL_B)
    for yy in np.linspace(cy - 0.9, cy + 0.9, 5):
        ax.annotate("", xy=(cx + 1.4, yy), xytext=(cx - 1.4, yy),
                    arrowprops=dict(arrowstyle="-|>", color=COL_A, lw=0.7, alpha=0.6))

    # right arrow + outcome
    ax.annotate("", xy=(8.2, 2.0), xytext=(7.5, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.text(9.15, 3.3, "frequency-gated\nfield penetration", ha="center",
            fontsize=7, color="#7a2016", weight="bold")
    ax.text(9.15, 2.35, r"$\bullet$ intracellular field", ha="center", fontsize=6.8)
    ax.text(9.15, 2.02, r"$\bullet$ dielectrophoresis", ha="center", fontsize=6.8)
    ax.text(9.15, 1.69, r"$\bullet$ mitotic disruption", ha="center", fontsize=6.8)
    ax.text(9.15, 1.36, r"$\bullet$ membrane permeab.", ha="center", fontsize=6.8)
    ax.add_patch(Rectangle((8.35, 1.2), 1.6, 1.35, fc="none", ec="#7a2016",
                           lw=0.8, ls="--"))

    fig.tight_layout()
    fig.savefig("figures/fig1_concept.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE 2 - dielectric dispersion of the cell: eps' and sigma vs freq
# ============================================================================
def fig2():
    epsp = np.array([shelled_sphere_eps(wi, 7.5e-6, d, eps_mem, sig_mem,
                                        eps_cyt, sig_cyt) for wi in w])
    eps_eff = np.real(epsp) / EPS0
    sig_eff = -np.imag(epsp) * w

    fig, ax1 = plt.subplots(figsize=(3.5, 2.9))
    shade_band(ax1)
    l1, = ax1.semilogx(freq, eps_eff, color=COL_A, lw=1.8)
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel(r"effective $\varepsilon'_{\rm cell}/\varepsilon_0$",
                   color=COL_A)
    ax1.tick_params(axis="y", labelcolor=COL_A)
    ax1.set_ylim(0, eps_eff.max() * 1.1)

    ax2 = ax1.twinx()
    l2, = ax2.semilogx(freq, sig_eff, color=COL_C, lw=1.8, ls="--")
    ax2.set_ylabel(r"effective $\sigma_{\rm cell}$ (S/m)", color=COL_C)
    ax2.tick_params(axis="y", labelcolor=COL_C)
    ax1.set_title(r"$\beta$-dispersion of a single-shell cell")
    ax1.legend([l1, l2], [r"$\varepsilon'$", r"$\sigma$"], loc="center left")
    fig.tight_layout()
    fig.savefig("figures/fig2_dispersion.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE 3 - intracellular field ratio + transmembrane potential vs freq
# ============================================================================
def fig3():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) intracellular field ratio for 3 cell sizes
    shade_band(axA)
    for R, c, ls in zip([5e-6, 10e-6, 20e-6], [COL_A, COL_B, COL_C],
                        ["-", "--", "-."]):
        ratio = np.array([core_field_ratio(wi, R, d, eps_mem, sig_mem,
                                           eps_cyt, sig_cyt, eps_med, sig_med)
                          for wi in w])
        axA.semilogx(freq, ratio, color=c, ls=ls, lw=1.7,
                     label=fr"$R={R*1e6:.0f}\,\mu$m")
    axA.set_xlabel("frequency (Hz)")
    axA.set_ylabel(r"$|E_{\rm in}/E_{\rm ext}|$")
    axA.set_title("(a) intracellular field penetration")
    axA.set_ylim(0, 1.25)
    axA.legend(loc="upper left")

    # (b) transmembrane potential vs freq
    shade_band(axB)
    for R, c, ls in zip([5e-6, 10e-6, 20e-6], [COL_A, COL_B, COL_C],
                        ["-", "--", "-."]):
        dVm, tau = transmembrane_potential(w, R, Cm, sig_cyt, sig_med, E0=100.0)
        axB.loglog(freq, dVm * 1e3, color=c, ls=ls, lw=1.7,
                   label=fr"$R={R*1e6:.0f}\,\mu$m")
    axB.set_xlabel("frequency (Hz)")
    axB.set_ylabel(r"$\Delta V_m$ (mV) at 1 V/cm")
    axB.set_title("(b) induced transmembrane potential")
    axB.legend(loc="lower left")
    axB.set_ylim(1e-2, 1e2)

    fig.tight_layout()
    fig.savefig("figures/fig3_field_penetration.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE 4 - Clausius-Mossotti factor (DEP) vs frequency
# ============================================================================
def fig4():
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    shade_band(ax)
    for sig_m, c, ls, lab in zip([0.3, 0.5, 1.0], [COL_A, COL_B, COL_C],
                                 ["-", "--", "-."],
                                 ["0.3", "0.5", "1.0"]):
        epsp = np.array([shelled_sphere_eps(wi, 7.5e-6, d, eps_mem, sig_mem,
                                            eps_cyt, sig_cyt) for wi in w])
        K = np.array([cm_factor(wi, ep, eps_med, sig_m)
                      for wi, ep in zip(w, epsp)])
        ax.semilogx(freq, np.real(K), color=c, ls=ls, lw=1.7,
                    label=fr"$\sigma_{{\rm med}}={lab}$ S/m")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"Re$[K(\omega)]$  (CM factor)")
    ax.set_title("Dielectrophoretic response")
    ax.set_ylim(-0.6, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig4_cm_factor.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE 5 - intracellular-field map vs cell radius and cytoplasm conductivity
# ============================================================================
def fig5():
    f0 = 200e3
    w0 = 2 * np.pi * f0
    R_arr = np.linspace(3e-6, 25e-6, 90)
    sig_arr = np.linspace(0.1, 0.8, 90)
    Z = np.zeros((len(sig_arr), len(R_arr)))
    for i, sc in enumerate(sig_arr):
        for j, R in enumerate(R_arr):
            Z[i, j] = core_field_ratio(w0, R, d, eps_mem, sig_mem,
                                       eps_cyt, sc, eps_med, sig_med)
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    pcm = ax.pcolormesh(R_arr * 1e6, sig_arr, Z, cmap="magma",
                        shading="auto", vmin=0, vmax=Z.max())
    cs = ax.contour(R_arr * 1e6, sig_arr, Z, levels=6, colors="w",
                    linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    ax.set_xlabel(r"cell radius $R$ ($\mu$m)")
    ax.set_ylabel(r"cytoplasm conductivity $\sigma_{\rm cyt}$ (S/m)")
    ax.set_title(r"$|E_{\rm in}/E_{\rm ext}|$ at 200 kHz")
    fig.colorbar(pcm, ax=ax, label=r"$|E_{\rm in}/E_{\rm ext}|$")
    fig.tight_layout()
    fig.savefig("figures/fig5_field_map.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE 6 - mitosis / dielectrophoresis schematic
# ============================================================================
def fig6():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax in (axA, axB):
        ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")

    # (a) interphase: near-uniform interior field, weak net force
    axA.set_title("(a) interphase: quasi-uniform field", fontsize=8.5)
    axA.add_patch(Circle((2.5, 2.5), 1.5, fc="#eef4fa", ec=COL_A, lw=2))
    for yy in np.linspace(1.2, 3.8, 6):
        axA.annotate("", xy=(4.6, yy), xytext=(0.4, yy),
                     arrowprops=dict(arrowstyle="-|>", color=COL_A, lw=0.8,
                                     alpha=0.55))
    axA.text(2.5, 2.5, r"$\vec E$", ha="center", va="center", fontsize=11,
             color=COL_A)

    # (b) telophase: hourglass -> field converges at furrow -> non-uniform
    axB.set_title("(b) cytokinesis: field focusing at furrow", fontsize=8.5)
    # two daughter lobes + narrow neck
    axB.add_patch(Circle((1.75, 2.5), 1.15, fc="#fdeeea", ec=COL_B, lw=2))
    axB.add_patch(Circle((3.25, 2.5), 1.15, fc="#fdeeea", ec=COL_B, lw=2))
    axB.add_patch(Rectangle((2.15, 2.15), 0.7, 0.7, fc="#fdeeea", ec="none"))
    # converging field lines toward neck
    for yy in np.linspace(1.5, 3.5, 7):
        axB.annotate("", xy=(2.5, 2.5), xytext=(0.3, yy),
                     arrowprops=dict(arrowstyle="-|>", color=COL_A, lw=0.7,
                                     alpha=0.5))
        axB.annotate("", xy=(4.7, yy), xytext=(2.5, 2.5),
                     arrowprops=dict(arrowstyle="-|>", color=COL_A, lw=0.7,
                                     alpha=0.5))
    # DEP force arrows toward high-field neck
    axB.annotate("", xy=(2.5, 2.5), xytext=(1.9, 2.5),
                 arrowprops=dict(arrowstyle="-|>", color=COL_C, lw=1.6))
    axB.annotate("", xy=(2.5, 2.5), xytext=(3.1, 2.5),
                 arrowprops=dict(arrowstyle="-|>", color=COL_C, lw=1.6))
    axB.text(2.5, 1.4, r"$\nabla|E|^2$ force on", ha="center", fontsize=7,
             color=COL_C)
    axB.text(2.5, 1.05, "polarisable organelles", ha="center", fontsize=7,
             color=COL_C)
    axB.text(2.5, 3.75, "cleavage\nfurrow", ha="center", fontsize=7,
             color=COL_B)

    fig.tight_layout()
    fig.savefig("figures/fig6_mitosis.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("All figures written to figures/")
