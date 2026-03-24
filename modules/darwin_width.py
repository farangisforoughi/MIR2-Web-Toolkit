import numpy as np
from modules.bragg import bragg_angle
from modules.structure_factor_si import structure_factor_si

# Physical / crystal constants in nm
R_E_NM = 2.8179403227e-6   # classical electron radius [nm]
A0_NM  = 0.54305           # Si lattice parameter [nm]
V_CELL_NM3 = A0_NM ** 3    # unit-cell volume [nm^3]


def chi_H_silicon(h: int, k: int, l: int, energy_keV: float, include_absorption: bool = True,) -> complex:
    
    res = bragg_angle(h, k, l, energy_keV)
    lam_nm = res["lambda_nm"]

    F_H = structure_factor_si(h, k, l, energy_keV, include_absorption=include_absorption)

    chi_H = - (R_E_NM * lam_nm**2 / (np.pi * V_CELL_NM3)) * F_H
    return chi_H


def darwin_half_width_symmetric(
    h: int,
    k: int,
    l: int,
    energy_keV: float,
    geometry: str = "bragg", include_absorption: bool = True,
) -> dict:
    
    res = bragg_angle(h, k, l, energy_keV)
    theta_B_rad = float(res["theta_rad"])

    geom = geometry.lower()
    if geom == "bragg":
        gamma0 = np.sin(theta_B_rad)
        gammaH = -np.sin(theta_B_rad)
    elif geom == "laue":
        gamma0 = np.sin(theta_B_rad)
        gammaH =  np.sin(theta_B_rad)
    else:
        raise ValueError(f"Unknown geometry: {geometry!r}. Use 'bragg' or 'laue'.")

    b = gamma0 / gammaH

    chi_H = chi_H_silicon(h, k, l, energy_keV, include_absorption=include_absorption)
    omega_rad = 2.0 * np.abs(chi_H) / (np.sqrt(np.abs(b)) * np.sin(2.0 * theta_B_rad))

    omega_urad = omega_rad * 1e6
    full_rad = 2.0 * omega_rad
    full_urad = 2.0 * omega_urad

    return {
        "omega_rad": float(omega_rad),
        "omega_urad": float(omega_urad),
        "full_rad": float(full_rad),
        "full_urad": float(full_urad),
        "theta_B_rad": float(theta_B_rad),
        "theta_B_deg": float(res["theta_deg"]),
        "gamma0": float(gamma0),
        "gammaH": float(gammaH),
        "b": float(b),
        "chi_H": chi_H,
        "include_absorption": bool(include_absorption),
    }
