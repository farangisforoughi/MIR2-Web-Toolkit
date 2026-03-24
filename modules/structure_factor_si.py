import numpy as np
from pathlib import Path

# -----------------------------
# Lattice parameter (Si, cubic)
# -----------------------------
A_SI = 5.4305  # Angstrom

# -----------------------------
# Cromer–Mann coefficients for Si
# -----------------------------
A1, A2, A3, A4 = 6.2915, 3.0353, 1.9891, 1.5410
B1, B2, B3, B4 = 2.4386, 32.3337, 0.6785, 81.6937
C_CM = 1.1407
z_Si = 14   #atomic number
def s_from_hkl(h: int, k: int, l: int, a_angstrom: float = A_SI) -> float:
    """
    Cromer–Mann scattering variable 
    k = sin(theta_B)/lambda for first-order Bragg reflection in a cubic crystal:
        k = sqrt(h^2 + k^2 + l^2) / (2 a)

    Returned in 1/Angstrom.
    """
    hklsq = h*h + k*k + l*l
    if hklsq == 0:
        return 0.0
    return float(np.sqrt(hklsq) / (2.0 * a_angstrom))


def f0_si(h: int, k: int, l: int) -> float:
    """
    Non-dispersive atomic form factor f0 for Si for the (hkl) reflection,
    using Cromer–Mann coefficients.
    """
    kval = s_from_hkl(h, k, l)  # [1/Å]
    k2 = kval * kval

    term1 = A1 * np.exp(-B1 * k2)
    term2 = A2 * np.exp(-B2 * k2)
    term3 = A3 * np.exp(-B3 * k2)
    term4 = A4 * np.exp(-B4 * k2)

    f0 = C_CM + term1 + term2 + term3 + term4
    return float(f0)


# -----------------------------
# NIST f1, f2 data for Si
# -----------------------------
def _load_nist_table(filename: str = "Si.txt") -> np.ndarray:
    path = Path(__file__).with_name(filename)
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                float(parts[0])
            except Exception:
                continue
            if len(parts) < 3:
                continue
            rows.append([float(x) for x in parts[:3]])

    if not rows:
        raise RuntimeError(f"No numeric data rows found in NIST file: {path}")
    return np.array(rows, dtype=float)


_nist = _load_nist_table()
E_NIST  = _nist[:, 0]
F1_NIST = _nist[:, 1]
F2_NIST = _nist[:, 2]


def diamond_phase_factor(h: int, k: int, l: int) -> complex:
    """
    Diamond-cubic lattice sum S_H(h,k,l).

    S_H = [1 + exp(i*pi/2*(h+k+l))] *
          [1 + exp(i*pi(h+k)) + exp(i*pi(k+l)) + exp(i*pi(h+l))]
    """
    h = int(h)
    k = int(k)
    l = int(l)

    t1 = 1.0 + np.exp(0.5j * np.pi * (h + k + l))
    t2 = (
        1.0
        + np.exp(1j * np.pi * (h + k))
        + np.exp(1j * np.pi * (k + l))
        + np.exp(1j * np.pi * (h + l))
    )
    return t1 * t2


def atomic_scattering_factor_si(
    h: int,
    k: int,
    l: int,
    energy_keV: float,
    include_dispersion: bool = True,
    include_absorption: bool = True,
) -> complex:
    """
    Atomic scattering factor for Si for the (hkl) reflection:

        f_Si(E, theta_H) ≈ f0(h,k,l) + f'(E) + i f''(E),

    where f'(E) = f1_NIST(E) - Z and f''(E) = f2_NIST(E).
    """
    f0 = f0_si(h, k, l)

    if include_dispersion:
        f1_total = float(np.interp(energy_keV, E_NIST, F1_NIST))  
        f1_corr  = f1_total - z_Si                               
    else:
        f1_corr = 0.0

    if include_absorption:
        f2 = float(np.interp(energy_keV, E_NIST, F2_NIST))       
    else:
        f2 = 0.0

    return f0 + f1_corr + 1j * f2


def structure_factor_si(
    h: int,
    k: int,
    l: int,
    energy_keV: float,
    include_absorption: bool = True,
) -> complex:
    """
    Unit-cell structure factor F_H(E) for silicon:

        F_H(E) = f_Si(E, theta_H) * S_H(h,k,l)

    If the reflection is systematically extinct, S_H = 0 and F_H = 0.
    """
    S_H = diamond_phase_factor(h, k, l)
    if S_H == 0:
        return 0.0 + 0.0j

    f_atom = atomic_scattering_factor_si(
        h, k, l, energy_keV,
        include_dispersion=True,
        include_absorption=include_absorption,
    )
    return f_atom * S_H


def structure_factor_si_no_absorption(h: int, k: int, l: int, energy_keV: float) -> complex:
    """
    Structure factor without the imaginary (absorptive) part:

        F_H^noabs(E) = [f0(h,k,l) + f1(E)] * S_H(h,k,l).
    """
    return structure_factor_si(h, k, l, energy_keV, include_absorption=False)
