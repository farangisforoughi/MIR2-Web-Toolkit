from __future__ import annotations
from typing import Optional

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import imageio.v3 as iio
from natsort import natsorted
from scipy.optimize import curve_fit

from modules.reflectivity_si import reflectivity_profile_symmetric_si, double_crystal_rocking_curve_si
from modules.bragg import is_allowed_silicon
# -------------------------------------------------------------------------
# Gaussian functions
# -------------------------------------------------------------------------

def Gaussian(x: np.ndarray, a: float, x0: float, wid: float) -> np.ndarray:
    """
        Gaussian(x) = a * exp( - (x - x0)^2 / (2 * wid^2) )
    """
    wid = np.abs(wid)
    z = (x - x0) / wid
    y = a * np.exp(-z**2 / 2.0)
    return y

# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def load_stack_from_dir(
    directory: str | Path,
    n_angles: int | None = None,
    allow_empty: bool = False,
) -> np.ndarray | None:
    directory = Path(directory)

    flist = natsorted(directory.glob("*.tif*"), key=lambda p: p.name.lower())
    if not flist:
        if allow_empty:
            return None
        raise FileNotFoundError(f"No .tif/.tiff files found in: {directory}")

    if n_angles is not None:
        flist = flist[:int(n_angles)]

    imgs = []
    for f in flist:
        im = iio.imread(str(f))
        # if TIFF comes as (ny,nx,channels), take first channel
        if im.ndim == 3:
            im = im[..., 0]
        imgs.append(im)

    stack = np.stack(imgs, axis=0).astype(np.float64)  # (N, Ny, Nx)
    return stack

_UNIT_TO_RAD = {
    "rad": 1.0,
    "mrad": 1e-3,
    "urad": 1e-6,
    "deg": np.pi / 180.0,
    "arcsec": np.pi / (180.0 * 3600.0),
}

def load_rc_from_text_bytes(file_bytes: bytes, unit: str) -> tuple[np.ndarray, np.ndarray]:

    if file_bytes is None:
        raise ValueError("No RC bytes provided for upload mode.")

    if unit not in _UNIT_TO_RAD:
        raise ValueError(f"Unknown angle unit {unit!r}. Choose from: {list(_UNIT_TO_RAD)}")

    txt = file_bytes.decode("utf-8", errors="ignore")
    rows = []
    for ln in txt.splitlines():
        s = ln.strip()
        if (not s) or s.startswith("#"):
            continue
        s = s.replace(",", " ")
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except Exception:
            continue

    if len(rows) < 3:
        raise ValueError("Uploaded RC must contain at least 3 numeric rows with two columns: angle intensity.")

    arr = np.asarray(rows, dtype=np.float64)
    ang = arr[:, 0] * _UNIT_TO_RAD[unit]   # radians
    I = arr[:, 1].astype(np.float64)

    # sort by angle
    idx = np.argsort(ang)
    ang = ang[idx]
    I = I[idx]

    I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
    I[I < 0] = 0.0

    angles_urad = ang * 1e6
    return angles_urad, I

def compute_ideal_rc_from_upload(file_bytes: bytes, unit: str = "urad") -> tuple[np.ndarray, np.ndarray]:
    """Build (angles_urad, pRC) from an uploaded ideal RC.
    """
    rc_angles_urad, I = load_rc_from_text_bytes(file_bytes, unit=unit)
    nrm = normalize_1d(I)

    a0 = 1.0
    x0 = float(rc_angles_urad[np.argmax(nrm)])
    wid0 = float(np.std(rc_angles_urad)) if rc_angles_urad.size > 1 else 1.0
    p0 = [a0, x0, max(wid0, 1e-12)]

    pRC, _ = curve_fit(Gaussian, rc_angles_urad, nrm, p0=p0, maxfev=200_000)
    pRC[2] = abs(pRC[2])
    return rc_angles_urad, pRC
# -------------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------------
def normalize_1d(x: np.ndarray) -> np.ndarray:
    """Normalize 1D array to [0, 1]."""
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)

def fit_gaussian_1d(
    x: np.ndarray,
    y: np.ndarray,
    p0: List[float],
    maxfev: int = 100_000,
) -> np.ndarray:
    """Fit a single Gaussian Gaussian(x, a, x0, wid) to 1D data."""
    params, _ = curve_fit(Gaussian, x, y, p0=p0, maxfev=maxfev)
    return params

def fit_gaussian_per_pixel(
    angles: np.ndarray,
    stack: np.ndarray,
    p0: List[float],
    maxfev: int = 1_000_000,
) -> np.ndarray:
    
    n_angles, ny, nx = stack.shape
    assert angles.shape[0] == n_angles

    data_ref = stack.reshape(n_angles, ny * nx)
    params = np.zeros((ny * nx, 3), dtype=np.float64)

    for idx in range(ny * nx):
        y = data_ref[:, idx]
        try:
            p, _ = curve_fit(Gaussian, angles, y, p0=p0, maxfev=maxfev)
            p[2] = abs(p[2])  
        except Exception:
            # if fit fails: zero amplitude, small positive width
            p = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        params[idx] = p

    return params.reshape(ny, nx, 3)


def choose_dark_for_stack(raw_dark: Optional[np.ndarray], n_target: int, shape2d: tuple[int, int]) -> np.ndarray:
    
    if raw_dark is None:
        return np.zeros(shape2d, dtype=np.float64)

    if raw_dark.ndim == 2:
        return raw_dark

    if raw_dark.ndim != 3:
        raise ValueError(f"raw_dark must be 2D or 3D, got shape {raw_dark.shape}")

    nd = raw_dark.shape[0]
    if nd == n_target:
        return raw_dark
    if nd == 1:
        return raw_dark[0]

    return np.median(raw_dark, axis=0)


# -------------------------------------------------------------------------
# Rocking-curve model from reflectivity_si
# -------------------------------------------------------------------------

def compute_ideal_rc_from_reflectivity(
    h: int,
    k: int,
    l: int,
    energy_keV: float,
    geometry: str,
    thickness_mm: float,
    n_points: int = 801,
    range_factor: float = 4.0,
    include_absorption: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    geom = geometry.lower()
    if geom != "bragg":
        raise ValueError(f"compute_ideal_rc_from_reflectivity supports only 'bragg' (symmetric). Got {geometry!r}")

    res = reflectivity_profile_symmetric_si(
        h=h,
        k=k,
        l=l,
        energy_keV=energy_keV,
        geometry=geom,
        thickness_mm=thickness_mm,
        n_points=int(n_points),
        range_factor=range_factor,
        include_absorption=include_absorption,
    )

    # Use the same angular grid as reflectivity_profile_symmetric_si
    theta_off_rad  = np.asarray(res["theta_offset_rad"], dtype=np.float64)
    rc_angles_urad = theta_off_rad * 1e6
    R_single       = np.asarray(res["R_single"], dtype=np.float64)

    rc_out = double_crystal_rocking_curve_si(theta_off_rad, R_single)
    if isinstance(rc_out, dict):
        RC = np.asarray(rc_out["rocking_curve"], dtype=np.float64)
    else:
        
        RC = np.asarray(rc_out, dtype=np.float64)

    nrmRC = normalize_1d(RC)

    # Fit Gaussian to normalized RC vs angle (µrad)
    a0 = 1.0
    x0 = float(rc_angles_urad[np.argmax(nrmRC)])
    wid0 = float(np.std(rc_angles_urad)) if rc_angles_urad.size > 1 else 1.0
    p0 = [a0, x0, max(wid0, 1e-12)]

    pRC, _ = curve_fit(Gaussian, rc_angles_urad, nrmRC, p0=p0, maxfev=200_000)

    return rc_angles_urad, pRC

# -------------------------------------------------------------------------
# Angular calibration from ref and object boxes
# -------------------------------------------------------------------------

def estimate_ref_angles(
    ref_stack: np.ndarray,
    rc_angles: np.ndarray,
    pRC: np.ndarray,
    calib_box: Tuple[slice, slice],
) -> tuple[np.ndarray, np.ndarray]:

    n_angles = ref_stack.shape[0]
    a = np.arange(n_angles, dtype=np.float64)
    a = a - np.mean(a)

    sy, sx = calib_box
    flt_box = np.sum(ref_stack[:, sy, sx], axis=(1, 2))
    nrmF = normalize_1d(flt_box)

    
    a0 = 1.0
    x0 = float(a[np.argmax(nrmF)])
    wid0 = float(np.std(a)) if a.size > 1 else 1.0
    p0 = [a0, x0, max(wid0, 1.0)]
    
    pF, _ = curve_fit(Gaussian, a, nrmF, p0=p0, maxfev=200_000)
    
    fangl = (pRC[2] / pF[2]) * (a - pF[1])

   
    a0 = 1.0
    x0 = float(fangl[np.argmax(nrmF)])
    wid0 = float(np.std(fangl)) if fangl.size > 1 else 1e-6
    p0 = [a0, x0, max(wid0, 1e-6)]
    
    pFang, _ = curve_fit(Gaussian, fangl, nrmF, p0=p0, maxfev=200_000)
    return fangl, pFang


def estimate_object_angles(
    obj_stack: np.ndarray,
    rc_angles: np.ndarray,
    pRC: np.ndarray,
    calib_box: Tuple[slice, slice],
) -> tuple[np.ndarray, np.ndarray]:

    n_angles = obj_stack.shape[0]
    a = np.arange(n_angles, dtype=np.float64)
    a = a - np.mean(a)

    sy, sx = calib_box
    img_box = np.sum(obj_stack[:, sy, sx], axis=(1, 2))
    nrmI = normalize_1d(img_box)

    a0 = 1.0
    x0 = float(a[np.argmax(nrmI)])
    wid0 = float(np.std(a)) if a.size > 1 else 1.0
    p0 = [a0, x0, max(wid0, 1.0)]
    
    pI, _ = curve_fit(Gaussian, a, nrmI, p0=p0, maxfev=200_000)
    iangl = (pRC[2] / pI[2]) * (a - pI[1])

    a0 = 1.0
    x0 = float(iangl[np.argmax(nrmI)])
    wid0 = float(np.std(iangl)) if iangl.size > 1 else 1e-6
    p0 = [a0, x0, max(wid0, 1e-6)]
    pIang, _ = curve_fit(Gaussian, iangl, nrmI, p0=p0, maxfev=200_000)
    return iangl, pIang


# -------------------------------------------------------------------------
# Main MIR2 per-slice analysis
# -------------------------------------------------------------------------

def mir2_bragg_single_slice(
    raw_dark: np.ndarray,
    raw_ref: np.ndarray,
    raw_obj: np.ndarray,
    crop_roi: Tuple[slice, slice],
    calib_box: Tuple[slice, slice],
    h: int,
    k: int,
    l: int,
    energy_keV: float,
    geometry: str = "bragg",
    thickness_mm: float = 2.0,
    rc_range_factor: float = 4.0,
    rc_n_points: int = 801,
    include_absorption_rc: bool = True,
    rc_source: str = "Si model",
    rc_upload_bytes: bytes | None = None,
    rc_upload_unit: str = "urad",
    maxfev_pixel: int = 200_000,
) -> Dict[str, np.ndarray]:

    if rc_source.lower().startswith("si"):
        if not is_allowed_silicon(h, k, l):
            raise ValueError(f"Si({h}{k}{l}) is systematically forbidden (diamond structure).")
    

    geom = geometry.lower()
    if geom != "bragg":
        raise ValueError(f"Only symmetric Bragg is implemented here. Got {geometry!r}")

    if raw_ref.shape[1:] != raw_obj.shape[1:]:
        raise ValueError(
            f"ref and object must have same (Ny,Nx). Got ref {raw_ref.shape[1:]}, obj {raw_obj.shape[1:]}."
        )

    n_ref = int(raw_ref.shape[0])
    n_obj = int(raw_obj.shape[0])

    sy, sx = crop_roi

    # choose dark separately for each stack length
    shape2d = raw_ref.shape[1:]  # (Ny,Nx)
    darkR = choose_dark_for_stack(raw_dark, n_ref, shape2d)
    darkO = choose_dark_for_stack(raw_dark, n_obj, shape2d)

    # crop + subtract (broadcast if 2D)
    if darkR.ndim == 3:
        darkR_crop = darkR[:, sy, sx]
    else:
        darkR_crop = darkR[sy, sx]

    if darkO.ndim == 3:
        darkO_crop = darkO[:, sy, sx]
    else:
        darkO_crop = darkO[sy, sx]

    ref_crop = (raw_ref[:, sy, sx] - darkR_crop).clip(min=1.0)
    obj_crop = (raw_obj[:, sy, sx] - darkO_crop).clip(min=1.0)

    _, ny, nx = ref_crop.shape  # define from ref (they match anyway)

    # 1) Ideal RC (µrad)
    if rc_source.lower().startswith("si"):
        rc_angles_urad, pRC = compute_ideal_rc_from_reflectivity(
            h=h, k=k, l=l,
            energy_keV=energy_keV,
            geometry=geom,
            thickness_mm=thickness_mm,
            n_points=int(rc_n_points),
            range_factor=rc_range_factor,
            include_absorption=include_absorption_rc,
        )
    else:
        rc_angles_urad, pRC = compute_ideal_rc_from_upload(
            file_bytes=rc_upload_bytes,
            unit=rc_upload_unit,
        )

    # 2) Estimate angles for each stack independently (µrad)
    fangl, _ = estimate_ref_angles(ref_crop, rc_angles_urad, pRC, calib_box)
    iangl, _ = estimate_object_angles(obj_crop, rc_angles_urad, pRC, calib_box)

    # 3) Initial guesses
    syb, sxb = calib_box
    Fbox = np.sum(ref_crop[:, syb, sxb], axis=(1, 2))
    Ibox = np.sum(obj_crop[:, syb, sxb], axis=(1, 2))

    p0F = [float(np.max(Fbox)), float(fangl[np.argmax(Fbox)]), float(np.std(fangl) + 1e-6)]
    p0I = [float(np.max(Ibox)), float(iangl[np.argmax(Ibox)]), float(np.std(iangl) + 1e-6)]

    # 4) Per-pixel Gaussian fits (outputs x0 and wid in µrad)
    params_F = fit_gaussian_per_pixel(fangl, ref_crop, p0=p0F, maxfev=maxfev_pixel)
    params_I = fit_gaussian_per_pixel(iangl, obj_crop, p0=p0I, maxfev=maxfev_pixel)

    A_F   = params_F[:, :, 0]
    x0_F  = params_F[:, :, 1]
    sig_F = np.abs(params_F[:, :, 2])

    A_I   = params_I[:, :, 0]
    x0_I  = params_I[:, :, 1]
    sig_I = np.abs(params_I[:, :, 2])

    # 5) Channels
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(A_F > 0, A_I / A_F, 0.0)

    # refraction in µrad
    refraction_urad = x0_I - x0_F

    sig_I_clp = sig_I.clip(min=1e-9)
    sig_F_clp = sig_F.clip(min=1e-9)

    # USAXS 
    usaxs_urad = np.sqrt((sig_I_clp**2 - sig_F_clp**2).clip(min=0.0))

    # absorption
    areaI = A_I * sig_I_clp * np.sqrt(2.0 * np.pi)
    areaF = A_F * sig_F_clp * np.sqrt(2.0 * np.pi)

    with np.errstate(divide="ignore", invalid="ignore"):
        transmission = np.where(areaF > 0, areaI / areaF, 1.0)
        radiograph = -np.log(transmission.clip(min=1e-12))  
        

    return {
        # maps
        "refraction": refraction_urad,
        "USAXS": usaxs_urad,
        "transmission": transmission,
        "radiograph": radiograph,
        "widI": sig_I_clp,
        "widF": sig_F_clp,
        "amp": amp,

        # curves/angles for plotting
        "ref_angl": fangl,
        "obj_angl": iangl,
        "ref_box_sum": Fbox,
        "obj_box_sum": Ibox,
    }
