import numpy as np
from pathlib import Path

# -----------------------------
# Physical and crystal constants
# -----------------------------
a0 = 0.54305                # Si lattice parameter [nm]
V_cell = a0**3              # conventional unit-cell volume [nm^3]

hc = 1.23984198             # Planck*c [keV·nm]
r_e = 2.8179403227e-6       # classical electron radius [nm]

rho_Si = 2.32               # density of Si [g/cm^3]


def _load_nist_table(filename: str = "Si.txt") -> np.ndarray:
    
    path = Path(__file__).with_name(filename)
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Try to parse first token as energy [keV]
            try:
                float(parts[0])
            except Exception:
                continue
            # Data lines should have at least 8 numeric columns
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])

    if not rows:
        raise RuntimeError(f"No numeric data rows found in NIST file: {path}")

    return np.array(rows, dtype=float)


# Load NIST table 
_nist_data = _load_nist_table()
E_NIST        = _nist_data[:, 0]   # keV
f1_NIST       = _nist_data[:, 1]   # e/atom
f2_NIST       = _nist_data[:, 2]   # e/atom
mu_photo_mass = _nist_data[:, 3]   # (mu/rho)_photo [cm^2/g]
muT_mass      = _nist_data[:, 5]   # (mu/rho)_total [cm^2/g]
lambda_NIST   = _nist_data[:, 7]   # nm 


def chi0_from_f1f2(energy_keV: float) -> complex:
    # wavelength in nm
    lam = hc / energy_keV

    # Linear interpolation of f1,f2 from NIST grid
    f1 = float(np.interp(energy_keV, E_NIST, f1_NIST))
    f2 = float(np.interp(energy_keV, E_NIST, f2_NIST))

    # Forward unit-cell structure factor: 8 Si atoms per cell
    F0 = 8.0 * (f1 + 1j * f2)  # electrons per unit cell

    # chi_0 from dynamical diffraction theory (Zachariasen)
    chi0 = - (r_e * lam**2 / (np.pi * V_cell)) * F0
    return chi0


def chi0_im_from_mu(energy_keV: float, use_total: bool = False) -> float:

    lam = hc / energy_keV  # [nm]

    if use_total:
        mu_mass = float(np.interp(energy_keV, E_NIST, muT_mass))       # [cm^2/g]
    else:
        mu_mass = float(np.interp(energy_keV, E_NIST, mu_photo_mass))  # [cm^2/g]

    mu_linear_cm = mu_mass * rho_Si      # [1/cm]
    mu_linear_nm = mu_linear_cm * 1e-7   # [1/nm]

    chi0_im = - mu_linear_nm * lam / (2.0 * np.pi)
    return chi0_im



def chi0_silicon(energy_keV: float) -> complex:
    
    return chi0_from_f1f2(energy_keV)


chi0_silicon_vec = np.vectorize(chi0_silicon)


if __name__ == "__main__":
    for E in [5.0, 10.0, 20.0, 30.0]:
        chi = chi0_silicon(E)
        chi_im_mu = chi0_im_from_mu(E)
        print(f"E = {E:5.1f} keV")
        print(f"  chi0      = {chi}")
        print(f"  Re(chi0)  = {chi.real:.3e}")
        print(f"  Im(chi0)  = {chi.imag:.3e}")
        print(f"  Im from μ = {chi_im_mu:.3e}")
