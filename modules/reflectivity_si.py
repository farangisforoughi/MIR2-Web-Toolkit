# crystal reflectivity and double-crystal rocking curve
# for symmetric Si reflections


import numpy as np
from modules.bragg import bragg_angle
from modules.chi0_silicon import chi0_silicon
from modules.structure_factor_si import structure_factor_si_no_absorption
from modules.darwin_width import chi_H_silicon, darwin_half_width_symmetric


a0 = 0.54305                # Si lattice parameter [nm]
V_cell = a0**3              # conventional unit-cell volume [nm^3]
r_e = 2.8179403227e-6       # classical electron radius [nm]

def reflectivity_profile_symmetric_si(
    h: int,
    k: int,
    l: int,
    energy_keV: float,
    geometry: str = "bragg",
    thickness_mm: float = 1.0,
    n_points: int = 401,
    range_factor: float = 4.0,
    include_absorption: bool = True,
    *,
    A_thin: float = 0.4,         # "A << 1" cutoff
    A_thick: float = 500,        # "A >> 1" cutoff 
):
    #-----------------------------
    # Bragg angle and wavelength
    # -----------------------------
    ba = bragg_angle(h, k, l, energy_keV)
    theta_B = float(ba["theta_rad"])
    lam_nm = float(ba["lambda_nm"])

    # χ0(E) from NIST f1, f2 
    chi0 = chi0_silicon(energy_keV)

    # χ_H(E) from structure factor
    chiH = chi_H_silicon(h, k, l, energy_keV)

    if not include_absorption:
        chi0 = complex(chi0.real, 0.0)
        chiH = complex(chiH.real, 0.0)

    # -----------------------------
    # Geometry factors
    # -----------------------------
    geom = geometry.lower().strip()
    if geom == "bragg":
        gamma0 = np.sin(theta_B)
        gammaH = -np.sin(theta_B)
    elif geom == "laue":
        gamma0 = np.sin(theta_B)
        gammaH =  np.sin(theta_B)
    else:
        raise ValueError("geometry must be 'bragg' or 'laue'")

    b = gamma0 / gammaH  # symmetric: bragg b=-1, laue b=+1

    # Darwin half width 
    dw = darwin_half_width_symmetric(
        h, k, l, energy_keV,
        geometry=geom,
        include_absorption=include_absorption
    )
    omega = float(dw["omega_rad"])

    # Refraction-corrected center (real χ0)
    theta_I = theta_B + chi0.real * ((1.0 - b) / (2.0 * b * np.sin(2.0 * theta_B)))

    # -----------------------------
    # Angular grid
    # -----------------------------
    if n_points % 2 == 0:
        n_points += 1

    dtheta = np.linspace(-range_factor * omega, +range_factor * omega, n_points)
    theta = theta_I + dtheta

    # rotating-crystal α ≈ 2(θ_B - θ) sin(2θ_B)
    alpha = 2.0 * (theta_B - theta) * np.sin(2.0 * theta_B)

    # -----------------------------
    # Thickness + reference rotation for χH
    # -----------------------------
    t0_nm = thickness_mm * 1.0e6  # mm -> nm

    # Reference (no-absorption) χH0 phase for this hkl (to define ψ'_H real)
    F0 = structure_factor_si_no_absorption(h, k, l, energy_keV)
    chiH0 = - (r_e * lam_nm**2 / (np.pi * V_cell)) * F0
    phi0 = np.angle(chiH0) if np.abs(chiH0) > 0 else 0.0

    chiH_rot = chiH * np.exp(-1j * phi0)

    # |psi'_H|: magnitude of the real coupling term in rotated frame
    eps = 1e-30
    chiH_realref = float(np.abs(chiH_rot.real))
    chiH_realref = max(chiH_realref, eps)

    # κ: ratio of imaginary to real coupling in rotated frame
    kappa = float(chiH_rot.imag / chiH_realref)

    # μ0 from Im(χ0): μ0 = -2π Im(χ0)/λ  [1/nm]
    mu0 = float(-(2.0 * np.pi * chi0.imag) / lam_nm)

    # z(θ) = 1/2 (1-b) χ0 + 1/2 b α  (complex if absorption)
    z = 0.5 * (1.0 - b) * chi0 + 0.5 * b * alpha

    # Reduced variables: y from Re(z), g from Im(z)
    scale = np.sqrt(np.abs(b)) * chiH_realref 
    y = z.real / scale
    g = z.imag / scale

    # Thickness parameter A: A = π t0 |ψ'_H| / (λ |γ0|) for symmetric case, K=1
    A = (np.pi * t0_nm * chiH_realref) / (np.abs(gamma0) * lam_nm)

    denom = 1.0 + y**2

    # -----------------------------
    # NO ABSORPTION: regime switch by A
    # -----------------------------
    if not include_absorption:
        if A <= A_thin:
            # Thin-crystal approximation: sin^2(A y) / y^2  (both Bragg & Laue)
            Ay = A * y
            R_single = np.empty_like(y, dtype=np.float64)
    
            mask0 = np.isclose(y, 0.0, atol=1e-14)
            maskn = ~mask0
            R_single[mask0] = A**2
            R_single[maskn] = (np.sin(Ay[maskn])**2) / (y[maskn]**2)
    
        elif A_thin < A < A_thick:
            # Exact plate formulas 
            if geom == "laue":
                R_single = (np.sin(A * np.sqrt(denom))**2) / denom
            else:
                R_single = np.empty_like(y, dtype=np.float64)
    
                mask_pos = (np.abs(y) > 1.0)   # |y|>1 -> cot^2
                mask_neg = (np.abs(y) < 1.0)   # |y|<1 -> coth^2
                mask_one = ~(mask_pos | mask_neg)
    
                if np.any(mask_pos):
                    yyp = y[mask_pos]
                    s = np.sqrt(np.clip(yyp**2 - 1.0, 0.0, None))
                    cot2 = (1.0 / np.tan(A * s))**2
                    R_single[mask_pos] = 1.0 / (yyp**2 + (yyp**2 - 1.0) * cot2)
    
                if np.any(mask_neg):
                    yyn = y[mask_neg]
                    s = np.sqrt(np.clip(1.0 - yyn**2, 0.0, None))
                    coth2 = (1.0 / np.tanh(A * s))**2
                    R_single[mask_neg] = 1.0 / (yyn**2 + (1.0 - yyn**2) * coth2)
    
                if np.any(mask_one):
                    R_single[mask_one] = 1.0
    
        else:
            # Thick mean (smoothed) curves
            if geom == "laue":
                R_single = 1.0 / (2.0 * (1.0 + y**2))  
            else:
                R_single = np.empty_like(y, dtype=np.float64)
                m_in = (np.abs(y) < 1.0)
                m_out = ~m_in
                R_single[m_in] = 1.0
                yy = y[m_out]
                R_single[m_out] = 1.0 - np.sqrt(np.maximum(0.0, 1.0 - 1.0/(yy**2)))  
    
        theta_offset_urad = dtheta * 1.0e6
        return {
            "theta_center_rad": theta_I,
            "theta_offset_rad": dtheta,
            "theta_offset_urad": theta_offset_urad,
            "theta_grid_rad": theta,
            "R_single": R_single,
            "A": float(A),
        }


    # -----------------------------
    # WITH ABSORPTION: thick absorbing formulas
    # -----------------------------
    if geom == "bragg":

        L = np.sqrt(
            (-1.0 + y**2 - g**2)**2 + 4.0*(g*y - kappa)**2
        ) + y**2 + g**2

        R_single = L - np.sqrt(L**2 - (1.0 + 4.0*kappa**2))

    else:
        #(thick absorbing Laue)
        sinh_term = np.sinh(kappa * A / denom)
        R_single = 0.5 * np.exp(-mu0 * t0_nm / gamma0) * (1.0 + 2.0 * sinh_term**2) / denom

    theta_offset_urad = dtheta * 1.0e6
    return {
        "theta_center_rad": theta_I,
        "theta_offset_rad": dtheta,
        "theta_offset_urad": theta_offset_urad,
        "theta_grid_rad": theta,
        "R_single": R_single,
        "A": float(A),
    }


def double_crystal_rocking_curve_si(
    theta_offset_rad: np.ndarray,
    R_single: np.ndarray | None = None,
    *,
    h: int | None = None,
    k: int | None = None,
    l: int | None = None,
    energy_keV: float | None = None,
    analyzer_geometry: str = "bragg",   
    thickness_mm: float = 1.0,
    n_points: int | None = None,
    range_factor: float = 4.0,
    include_absorption: bool = True,
) -> np.ndarray:
    """
    MIR-aware mode:
      - If analyzer_geometry="bragg": DCM Bragg, analyzer Bragg
      - If analyzer_geometry="laue":  DCM Bragg, analyzer Laue
      - Rocking curve = (R_DCM ⊗ R_analyzer), normalized by ∑ R_DCM.
    """

    if R_single is not None and h is None:
        R_dcm = R_single ** 2
        R_analyzer = R_single
        rc = np.convolve(R_dcm, R_analyzer, mode="same")
        norm = np.sum(R_dcm)
        if norm > 0:
            rc = rc / norm
        return {"R_analyzer": R_analyzer, "rocking_curve": rc}

    # ----------------------------
    # MIR-aware mode
    # ----------------------------
    if any(v is None for v in (h, k, l, energy_keV)):
        raise ValueError(
        )

    if n_points is None:
        n_points = int(theta_offset_rad.size)

    res_dcm = reflectivity_profile_symmetric_si(
        h=int(h), k=int(k), l=int(l),
        energy_keV=float(energy_keV),
        geometry="bragg",
        thickness_mm=float(thickness_mm),
        n_points=int(n_points),
        range_factor=float(range_factor),
        include_absorption=bool(include_absorption),
    )
    R_dcm_single = res_dcm["R_single"]
    R_dcm = R_dcm_single ** 2  # two Bragg reflections in the DCM

    # Analyzer geometry follows user selection ("bragg" or "laue")
    geom_an = analyzer_geometry.lower().strip()
    if geom_an not in ("bragg", "laue"):
        raise ValueError("analyzer_geometry must be 'bragg' or 'laue'")

    res_an = reflectivity_profile_symmetric_si(
        h=int(h), k=int(k), l=int(l),
        energy_keV=float(energy_keV),
        geometry=geom_an,
        thickness_mm=float(thickness_mm),
        n_points=int(n_points),
        range_factor=float(range_factor),
        include_absorption=bool(include_absorption),
    )
    R_analyzer = res_an["R_single"]

    # Convolution: system rocking curve
    rc = np.convolve(R_dcm, R_analyzer, mode="same")
    norm = np.sum(R_dcm)
    if norm > 0:
        rc = rc / norm

    return {"R_analyzer": R_analyzer, "rocking_curve": rc}


