import numpy as np

a0 = 0.54305
hc = 1.23984198

def is_allowed_silicon(h: int, k: int, l: int) -> bool:
    """
    Selection rules for Si (diamond cubic).
    Returns True if reflection (hkl) is allowed, False if systematically forbidden.
    """
    # Exclude trivial 000
    if h == 0 and k == 0 and l == 0:
        return False

    # Parity of each index
    ph, pk, pl = h % 2, k % 2, l % 2

    # 1) Mixed parity → forbidden
    if not (ph == pk == pl):
        return False

    # 2) All odd → allowed
    if ph == 1:
        return True

    # 3) All even → allowed only if h+k+l = 4n
    s = h + k + l
    return (s % 4 == 0)

def bragg_angle(h, k, l, energy_keV):
    wavelength = hc / energy_keV
    hkl_norm = np.sqrt(h**2 + k**2 + l**2)
    d_spacing = a0 / hkl_norm
    sin_th = wavelength / (2 * d_spacing)
    if sin_th > 1:
        raise ValueError("Energy too low for this reflection (sinθ > 1).")
    theta_rad = np.arcsin(sin_th)
    theta_deg = float(np.degrees(theta_rad))
    theta_urad = float(theta_rad * 1e6)
    return {
        "theta_rad": float(theta_rad),
        "theta_deg": theta_deg,
        "theta_urad": theta_urad,
        "d_nm": float(d_spacing),
        "lambda_nm": float(wavelength),
    }
