import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import io
import imageio.v3 as iio
from pathlib import Path
from modules.bragg import bragg_angle, is_allowed_silicon
from modules.chi0_silicon import chi0_silicon, chi0_im_from_mu, mu_photo_mass, muT_mass, rho_Si
from modules.structure_factor_si import (structure_factor_si,structure_factor_si_no_absorption,diamond_phase_factor,
    f0_si,atomic_scattering_factor_si,E_NIST, F1_NIST)
from modules.darwin_width import darwin_half_width_symmetric
from modules.reflectivity_si import (reflectivity_profile_symmetric_si, double_crystal_rocking_curve_si)
from modules.mir2_analysis import (load_stack_from_dir,mir2_bragg_single_slice,)


# ------------------------------------------
# STREAMLIT APP LAYOUT
# ------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="MIR2 Toolkit",
    layout="wide"
)

st.title("🔬 MIR2 Web Toolkit")
st.write("Web-based scientific tools for analyzer-based X-ray imaging (Only for Si crystals).")

# ------------------------------------------
# TAB LAYOUT
# ------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
    "Bragg Angle (Si)",
    "Forward χ₀(E) (Si)",
    "Structure Factor Fₕ (Si)",
    "Darwin Width (symmetric, σ)",
    "Reflectivity & Rocking Curve",
    "MIR2 Analysis",
    "About & Citation",
])

# ------------------------------------------
# helper function to save MIR2 results as TIFFs
# ------------------------------------------
def save_mir2_results_tiff(result: dict, out_dir: str, prefix: str = "mir2") -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    keys = ["refraction", "USAXS", "radiograph", "transmission", "widI", "widF"]
    for k in keys:
        if k not in result:
            continue
        arr = np.asarray(result[k], dtype=np.float64)
        iio.imwrite(str(outp / f"{k}.tif"), arr)
    
# ==========================================
# TAB 1 — BRAGG ANGLE
# ==========================================
with tab1:
    st.header("Bragg Angle Calculator")
    st.write(
        "Compute the Bragg angle for silicon (diamond-cubic) analyzer crystals. "
        "Selection rules for Si are applied to warn about systematically forbidden reflections."
    )

    col1, col2 = st.columns([1,1] )

    with col1:
        # ------ inputs parameters ------        
        st.subheader("Input Parameters")
        h_input,k_input,l_input = st.columns(3)
        h = h_input.number_input("h", value=2, step=1)
        k = k_input.number_input("k", value=2, step=1)
        l = l_input.number_input("l", value=0, step=1)
        
        E = st.number_input("X-ray Energy (keV)", value=30.0, step=0.1)

        # ------ calculation and output ------
        st.subheader("Results")

        hi, ki, li = int(h), int(k), int(l)

        allowed = is_allowed_silicon(hi, ki, li)
        if not allowed:
            st.warning(
                f"Reflection ({hi}{ki}{li}) is systematically forbidden for Si (diamond structure). "
                "The Bragg angle is still defined geometrically, but the ideal crystal reflectivity is zero."
            )

        try:
            result = bragg_angle(hi, ki, li, E)
            st.success("Bragg-angle calculation successful.")

            st.write(f"**Bragg angle (degrees):** {result['theta_deg']:.6f}")
            st.write(f"**Bragg angle (radians):** {result['theta_rad']:.6e}")
            #st.write(f"**Bragg angle (µrad):** {result['theta_urad']:.2f}")
            st.write(f"**Wavelength (nm):** {result['lambda_nm']:.6f}")
            st.write(f"**d-spacing (nm):** {result['d_nm']:.6f}")

        except Exception as e:
            st.error(f"Error in Bragg-angle calculation: {e}")

    #st.markdown("---")

    # 
    # ------ PLOT: Bragg angle vs energy ------
    with col2:

        hi, ki, li = int(h), int(k), int(l)
        E_vals = np.linspace(5, 80, 300)
        theta_vals = []

        for e in E_vals:
            try:
                theta_vals.append(bragg_angle(hi, ki, li, e)["theta_deg"])
            except Exception:
                theta_vals.append(np.nan)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(E_vals, theta_vals)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Bragg Angle (deg)")
        ax.set_title(f"Bragg Angle vs Energy for Si({hi}{ki}{li})")
        ax.grid(True)
        st.pyplot(fig)


# ==========================================
# TAB 2 — FORWARD χ₀(E) FOR Si
# ==========================================
with tab2:
    st.header("Forward susceptibility χ₀(E) for silicon")
    st.write(
        "Compute the forward Fourier component χ₀(E) for silicon from NIST f₁(E), f₂(E), "
        "and compare its imaginary part to values inferred from the NIST mass attenuation coefficients. "
        "Here χ₀ corresponds to the H = 0 Fourier component (forward / transmitted beam)."
    )

    
    # ------- single-energy evalulation ---------
    col1, col2 = st.columns([0.25, 0.75])

    with col1:
        E_single = st.number_input(
            "Energy for χ₀ evaluation (keV)",
            value=30.0,
            step=0.5,
            key="E_single_chi0"
        )
        
        st.subheader("Energy dependence of χ₀(E)")
        
        # input numbers
        E_min = st.number_input("Energy range min (keV)", value=1.0, step=1.0, key="E_min_chi0")
        E_max = st.number_input("Energy range max (keV)", value=40.0, step=1.0, key="E_max_chi0")
        n_points = st.slider("Number of points", min_value=50, max_value=400, value=200, step=10)
            
        try:
            chi = chi0_silicon(E_single)
            chi_im_photo = chi0_im_from_mu(E_single, use_total=False)
            chi_im_total = chi0_im_from_mu(E_single, use_total=True)

            st.success("χ₀(E) evaluation successful.")

            st.write(f"**χ₀(E)** = {chi.real:.3e} + {chi.imag:.3e} i")
            st.write(f"**Im[χ₀(E)] photoelectric** = {chi_im_photo:.6e}")
            st.write(f"**Im[χ₀(E)] total** = {chi_im_total:.6e}")

         

        except Exception as e:
            st.error(f"Error in χ₀ evaluation: {e}")


    col_input, col_plot1, col_plot2 = st.columns([0.2, 0.4, 0.4])

    with col2:
        # -------- Energy range plot --------
        if E_max <= E_min:
            st.error("Energy max must be greater than energy min.")
        else:
            # Build energy grid and χ₀ arrays ONCE
            E_grid = np.linspace(E_min, E_max, n_points)
            chi_real = np.zeros_like(E_grid)
            chi_imag = np.zeros_like(E_grid)
            chi_im_photo = np.zeros_like(E_grid)
            chi_im_total = np.zeros_like(E_grid)
            
            mu_photo_cm = np.zeros_like(E_grid)
            mu_total_cm = np.zeros_like(E_grid)
            

            for i, E_val in enumerate(E_grid):
                try:
                    chi_val = chi0_silicon(E_val)
                    chi_real[i] = chi_val.real
                    chi_imag[i] = chi_val.imag
                    chi_im_photo[i] = chi0_im_from_mu(E_val, use_total=False)
                    chi_im_total[i] = chi0_im_from_mu(E_val, use_total=True)
                    mu_photo_mass_val = float(np.interp(E_val, E_NIST, mu_photo_mass))  # [cm^2/g]
                    mu_total_mass_val = float(np.interp(E_val, E_NIST, muT_mass))       # [cm^2/g]
                    mu_photo_cm[i] = mu_photo_mass_val * rho_Si                         # [1/cm]
                    mu_total_cm[i] = mu_total_mass_val * rho_Si
                except Exception:
                    chi_real[i] = np.nan
                    chi_imag[i] = np.nan
                    chi_im_photo[i] = np.nan
                    chi_im_total[i] = np.nan
                    mu_photo_cm[i] = np.nan
                    mu_total_cm[i] = np.nan

            col_plot1, col_plot2 = st.columns(2)

            # Plot Re[χ₀(E)]
            with col_plot1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(E_grid, chi_real, color = "orange")
                ax1.set_xlabel("Energy (keV)")
                ax1.set_ylabel("Re[χ₀(E)]")
                ax1.set_title("Real part χ₀(E) for silicon")
                ax1.grid(True)
                st.pyplot(fig1)

            # Plot Im[χ₀(E)]
            with col_plot2:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(E_grid, chi_im_photo,  label="Im[χ₀] photoelectric", color = "g")
                ax2.set_xlabel("Energy (keV)")
                ax2.set_ylabel("Im[χ₀(E)]")
                ax2.set_title("Imaginary part of χ₀(E) for silicon")
                ax2.grid(True)
                #ax2.legend()
                st.pyplot(fig2)
                
                st.markdown("---")
                
        
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.plot(E_grid, mu_photo_cm, label="μ photoelectric", color = "b")
                #ax3.plot(E_grid, mu_total_cm, linestyle="--", label="μ total")
                ax3.set_xlabel("Energy (keV)")
                ax3.set_ylabel("μ (1/cm)")
                ax3.set_title("NIST-based Linear attenuation coefficient μ(E) for silicon")
                ax3.grid(True)
                #ax3.legend()
                st.pyplot(fig3)

# ==========================================
# TAB 3 — STRUCTURE FACTOR F_H(E) FOR Si
# ==========================================
with tab3:
    st.header("Structure factor F\u2095(E) for silicon")

    st.write(
        "Compute the unit-cell structure factor F\u2095(E) for Si using the diamond-lattice "
        "phase factor and the atomic scattering factor "
        "f(E, \u03b8) \u2248 f\u2080 + f\u2032(E) + i f\u2033(E). "
        "This uses the same Cromer\u2013Mann f\u2080 and NIST f\u2081, f\u2082 tables as in the \u03c7\u2080(E) module."
    )

    # -------- single-energy calculation --------
    col1, col2 = st.columns([0.8, 1.2])

    with col1:
        hcol, kcol, lcol = st.columns(3)
        h_sf = hcol.number_input("h (structure factor)", value=2, step=1, key="sf_h", min_value=0)
        k_sf = kcol.number_input("k (structure factor)", value=2, step=1, key="sf_k", min_value=0)
        l_sf = lcol.number_input("l (structure factor)", value=0, step=1, key="sf_l", min_value=0)
        
        E_sf = st.number_input("Photon energy (keV)", value=30.0, step=0.5, key="sf_E")

        h_int = int(h_sf)
        k_int = int(k_sf)
        l_int = int(l_sf)

        try:
            # lattice sum (diamond phase factor)
            S_H = diamond_phase_factor(h_int, k_int, l_int)
            S_mag = np.abs(S_H)
            S_phase = np.angle(S_H, deg=True)

            if S_mag < 1e-6:
                st.warning(
                    f"Reflection ({h_int}{k_int}{l_int}) is systematically forbidden for Si (diamond). "
                    "The lattice sum S\u2095(h,k,l) vanishes and the ideal crystal reflectivity is zero."
                )

            # per-atom pieces
            f0_val = f0_si(h_int, k_int, l_int)
            f_atom_full = atomic_scattering_factor_si(
                h_int, k_int, l_int, E_sf,
                include_dispersion=True,
                include_absorption=True,
            )
            f_atom_noabs = atomic_scattering_factor_si(
                h_int, k_int, l_int, E_sf,
                include_dispersion=True,
                include_absorption=False,
            )

            # NIST f1_total and dispersion corrections
            f1_total = float(np.interp(E_sf, E_NIST, F1_NIST))  # ≈ Z + f′(E)
            f1_corr = f_atom_noabs.real - f0_val  # f′(E)
            f2_val = f_atom_full.imag             # f″(E)

            # unit-cell structure factors
            F = structure_factor_si(h_int, k_int, l_int, E_sf, include_absorption=True)
            F_noabs = structure_factor_si_no_absorption(h_int, k_int, l_int, E_sf)

            mag_F = np.abs(F)
            phase_F = np.angle(F, deg=True)   #if z = x +iy , r = |z|, cos phi = x/r

            st.success("Structure factor evaluation successful.")

            st.write(f"**F\u2095(E)** = {F.real:.3e} + {F.imag:.3e} i")
            st.write(f"**|F\u2095(E)|** = {mag_F:.3e}")
            st.write(f"**arg(F\u2095(E))** = {phase_F:.2f} degrees")
            st.write(f"**F\u2095(E) without absorption** = {F_noabs.real:.3e} + {F_noabs.imag:.3e} i")

            # ---------- detailed debug info ----------
            with st.expander("Show detailed components (f₀, f′, f″, Sₕ, Fₕ)", expanded=False):
                st.markdown("**Per-atom scattering (Si):**")
                st.write(f"f₀(hkl)                = {f0_val:.6f}  (non-dispersive Thomson term)")
                st.write(f"f₁,total^NIST(E)       = {f1_total:.6f}  (≈ Z + f′(E))")
                st.write(f"f′(E) = f₁,total−Z     = {f1_corr:.6f}  (dispersive real correction)")
                st.write(f"f″(E) = f₂(E)          = {f2_val:.6f}  (dispersive imaginary correction)")
                st.write(f"f_atom(E)              = {f_atom_full.real:.6f} + {f_atom_full.imag:.6f} i")

                st.markdown("---")
                st.markdown("**Diamond lattice sum Sₕ(h,k,l):**")
                st.write(f"Sₕ(h,k,l)             = {S_H.real:.6f} + {S_H.imag:.6f} i")
                st.write(f"|Sₕ|                   = {S_mag:.6f}")
                st.write(f"arg(Sₕ)                = {S_phase:.2f} degrees")

                st.markdown("---")
                st.markdown("**Unit-cell structure factor:**")
                st.write(f"Fₕ(E)                  = {F.real:.6f} + {F.imag:.6f} i")
                st.write(f"|Fₕ(E)|                = {mag_F:.6f}")
                st.write(
                    f"|Fₕ(E)| / |Sₕ|         = "
                    f"{mag_F / S_mag if S_mag > 0 else np.nan:.6f}  (≈ per-atom amplitude)"
                )
            
        except Exception as e:
            st.error(f"Error in structure-factor calculation: {e}")
            
    with col2:
        # -------- energy scan plot --------
        sf_plot_container = st.container()
        sf_input_container = st.container()
        
        with sf_input_container:
            sf_col_emin, sf_col_emax, sf_col_npts = st.columns(3)
            E_min_sf = sf_col_emin.number_input("Energy range min (keV)", value=1.0, step=1.0, key="sf_Emin", min_value=0.0)
            E_max_sf = sf_col_emax.number_input("Energy range max (keV)", value=40.0, step=1.0, key="sf_Emax", min_value=0.0)
            n_points_sf = sf_col_npts.slider("Number of points", min_value=50, max_value=400, value=200, step=10, key="sf_Npts")
        
            E_grid_sf = np.linspace(E_min_sf, E_max_sf, n_points_sf)
            mag_F_arr = np.zeros_like(E_grid_sf)
            mag_F_noabs_arr = np.zeros_like(E_grid_sf)
        
        if E_max_sf <= E_min_sf:
            st.error("Energy max must be greater than energy min.")
        else:
            h_int = int(h_sf)
            k_int = int(k_sf)
            l_int = int(l_sf)

            S_H = diamond_phase_factor(h_int, k_int, l_int)
            if np.abs(S_H) < 1e-6:
                st.warning(
                    f"Reflection ({h_int}{k_int}{l_int}) is systematically forbidden. "
                    "F\u2095(E) = 0 over the entire energy range."
                )
            else:
                for i, E_val in enumerate(E_grid_sf):
                    try:
                        F_val = structure_factor_si(h_int, k_int, l_int, E_val, include_absorption=True)
                        F_noabs_val = structure_factor_si_no_absorption(h_int, k_int, l_int, E_val)
                        mag_F_arr[i] = np.abs(F_val)
                        mag_F_noabs_arr[i] = np.abs(F_noabs_val)
                    except Exception:
                        mag_F_arr[i] = np.nan
                        mag_F_noabs_arr[i] = np.nan
                        
        with sf_plot_container:
            pad_l, main, pad_r = st.columns([0.1, 0.8, 0.1])
            with main:
                # plot container
                fig_sf, ax_sf = plt.subplots(figsize=(6, 4))
                ax_sf.plot(E_grid_sf, mag_F_arr, label="|F\u2095(E)| with absorption")
                ax_sf.plot(E_grid_sf, mag_F_noabs_arr, linestyle="--", label="|F\u2095(E)| without absorption")
                ax_sf.set_xlabel("Energy (keV)", fontsize=10)
                ax_sf.set_ylabel("|F\u2095(E)| (electrons)", fontsize=10)
                ax_sf.set_title(f"Structure factor magnitude vs energy for Si({h_int}{k_int}{l_int})", fontsize=10)
                ax_sf.tick_params(axis='both', labelsize=9)
                ax_sf.grid(True)
                ax_sf.legend()
                st.pyplot(fig_sf)


# ==========================================
# TAB 4 — DARWIN WIDTH (SYMMETRIC, σ-POL)
# ==========================================
with tab4:
    st.header("Darwin angular width (symmetric, σ polarization)")

    st.write(
        "Compute the dynamical Darwin angular half-width ωₜₕ for a symmetric Si reflection "
        "using two-beam theory, assuming σ polarization. "
    )

    col1, col2 = st.columns([0.8, 1.2])

    # -------- single-energy calculation --------
    with col1:
        hcol, kcol, lcol = st.columns(3)
        h_dw = hcol.number_input("h (Darwin width)", value=2, step=1, key="dw_h", min_value=0)
        k_dw = kcol.number_input("k (Darwin width)", value=2, step=1, key="dw_k", min_value=0)
        l_dw = lcol.number_input("l (Darwin width)", value=0, step=1, key="dw_l", min_value=0)
        E_dw = st.number_input("Photon energy (keV)", value=30.0, step=0.5, key="dw_E")


        h_int = int(h_dw)
        k_int = int(k_dw)
        l_int = int(l_dw)
        geometry_code = "bragg"


        try:
            res_dw = darwin_half_width_symmetric(
                h_int, k_int, l_int, E_dw, geometry=geometry_code
            )

            st.success("Darwin-width calculation successful.")

            st.write(f"**Bragg angle θᴮ (deg):** {res_dw['theta_B_deg']:.6f}")
            st.write(f"**Bragg angle θᴮ (rad):** {res_dw['theta_B_rad']:.6e}")
            st.write(f"**γ₀:** {res_dw['gamma0']:.6e}")
            st.write(f"**γₕ:** {res_dw['gammaH']:.6e}")
            st.write(f"**b = γ₀/γₕ:** {res_dw['b']:.1e}")

            st.write(f"**Darwin half-width ωₜₕ (µrad):** {res_dw['omega_urad']:.3f}")

            chi_H = res_dw["chi_H"]
            with st.expander("Show χ_H(E) for this reflection", expanded=False):
                st.write(f"χ_H(E) = {chi_H.real:.3e} + {chi_H.imag:.3e} i")
                st.write(f"|χ_H(E)| = {np.abs(chi_H):.3e}")


        except Exception as e:
            st.error(f"Error in Darwin-width calculation: {e}")

    with col2:
        
        # -------- energy scan plot --------
    
        # create containers for inputs and plots
        dw_plot_container = st.container()
        dw_input_container = st.container()
        
        with dw_input_container:
            emin_input, emax_input, nopointslider = st.columns(3)
            E_min_dw = emin_input.number_input("Energy range min (keV)", value=1.0, step=1.0, key="dw_Emin")
            E_max_dw = emax_input.number_input("Energy range max (keV)", value=40.0, step=1.0, key="dw_Emax")
            n_points_dw = nopointslider.slider(
                "Number of points (Darwin width scan)",
                min_value=50, max_value=400, value=200, step=10, key="dw_Npts"
            )
            
        E_grid_dw = np.linspace(E_min_dw, E_max_dw, n_points_dw)
        omega_arr = np.zeros_like(E_grid_dw)
        full_arr = np.zeros_like(E_grid_dw)
            
        # compute data (outside input container)
        if E_max_dw <= E_min_dw:
            st.error("Energy max must be greater than energy min.")
        else:

            h_int = int(h_dw)
            k_int = int(k_dw)
            l_int = int(l_dw)
            geometry_code = "bragg"
            for i, E_val in enumerate(E_grid_dw):
                try:
                    res_val = darwin_half_width_symmetric(
                        h_int, k_int, l_int, E_val, geometry=geometry_code
                    )
                    omega_arr[i] = res_val["omega_urad"]
                    full_arr[i] = res_val["full_urad"]
                except Exception:
                    omega_arr[i] = np.nan
                    full_arr[i] = np.nan
                    
                    
                    
        with dw_plot_container:
            
            pad_l, main, pad_r = st.columns([0.1, 0.8, 0.1])
            with main:
                
                fig_dw, ax_dw = plt.subplots(figsize=(6,4), dpi=900)
                ax_dw.plot(E_grid_dw, omega_arr)
                ax_dw.set_xlabel("Energy (keV)", fontsize=10)
                ax_dw.set_ylabel("Width (µrad)", fontsize=10)
                ax_dw.set_title(f"Darwin width (ωₜₕ) vs energy for Si({h_int}{k_int}{l_int})", fontsize=10)
                ax_dw.grid(True)
                ax_dw.tick_params(axis='both', labelsize=9)
                
                st.pyplot(fig_dw)

       
# ==========================================
# TAB 5 — REFLECTIVITY & ROCKING CURVE
# ==========================================
with tab5:
    st.header("Single-crystal, double-crystal reflectivity and rocking curve")

    st.write(
        "Thick-crystal reflectivity for symmetric Si(hkl) in the Bragg or Laue case, "
        "using the dynamical theory of absorbing crystals. "
        "The double-crystal reflectivity is built from the single-crystal profile, "
        "and the rocking curve is the convolution of DCM and analyzer reflectivity."
    )

    col_inputs, col_plots = st.columns([0.25, 0.75])

    with col_inputs:
        h_rc = st.number_input("h (reflectivity)", value=2, step=1, key="rc_h")
        k_rc = st.number_input("k (reflectivity)", value=2, step=1, key="rc_k")
        l_rc = st.number_input("l (reflectivity)", value=0, step=1, key="rc_l")

        E_rc = st.number_input("Photon energy (keV)", value=30.0, step=0.5, key="rc_E")

        geom_choice = st.radio(
            "Geometry",
            options=["Symmetric Bragg", "Symmetric Laue"],
            index=0,
            horizontal=True,
            key="rc_geom"
        )

        thickness_mm = st.number_input(
            "Crystal thickness (mm), below 1 mm is considered thin crystal regime",
            value=20.0,
            min_value=0.0001,
            max_value=50.0,
            step=0.01,
            format="%.4f",
            key="rc_thickness"
        )

        range_factor = st.slider(
            "Angular range (in units of µrad)",
            min_value=2.0,
            max_value=8.0,
            value=4.0,
            step=0.5,
            key="rc_range_factor"
        )

        n_points_rc = st.slider(
            "Number of angular samples",
            min_value=101,
            max_value=801,
            value=401,
            step=50,
            key="rc_n_points"
        )

        include_abs = st.checkbox(
            "Include absorption",
            value=True,
            key="rc_include_abs"
        )

        analyzer_geom = "bragg" if "Bragg" in geom_choice else "laue"
        
        resB = reflectivity_profile_symmetric_si(
            h_rc, k_rc, l_rc, E_rc,
            geometry="bragg",
            thickness_mm=thickness_mm,
            n_points=n_points_rc,
            range_factor=range_factor,
            include_absorption=include_abs,
        )

        theta_off_urad = resB["theta_offset_urad"]
        theta_off_rad  = resB["theta_offset_rad"]
        R_single       = resB["R_single"]
        A              = float(resB.get("A", np.nan))
        R_dcm = R_single ** 2

        h_int, k_int, l_int = int(h_rc), int(k_rc), int(l_rc)

        # --- Rocking curve
        if analyzer_geom == "bragg":
            rc_out = double_crystal_rocking_curve_si(theta_off_rad, R_single)
        else:
            rc_out = double_crystal_rocking_curve_si(
                theta_off_rad,
                None,
                h=h_int, k=k_int, l=l_int,
                energy_keV=float(E_rc),
                analyzer_geometry="laue",
                thickness_mm=float(thickness_mm),
                n_points=int(n_points_rc),
                range_factor=float(range_factor),
                include_absorption=bool(include_abs),
            )
            
        if not isinstance(rc_out, dict):
            st.error(f"double_crystal_rocking_curve_si returned {type(rc_out)} (expected dict). Clear Streamlit cache.")
            st.stop()

        R_analyzer = rc_out["R_analyzer"]
        rc_curve   = rc_out["rocking_curve"]

        st.success("Reflectivity and rocking-curve calculation successful.")

        st.markdown("**Summary:**")
        st.write(f"Reflection: Si({h_int}{k_int}{l_int}) at {E_rc:.2f} keV")
        st.write(f"Thickness parameter A: {A:.3g}")
        
    with col_plots:
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.plot(theta_off_urad, R_analyzer, color="#1f77b4")
            ax1.set_xlabel("Angle offset Δθ (µrad)")
            ax1.set_ylabel("Reflectivity I/I₀")
            ax1.set_title(f"Analyzer crystal reflectivity ({analyzer_geom})")
            ax1.grid(True)
            st.pyplot(fig1)

        with row1_col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(theta_off_urad, R_dcm, color="#ff7f0e")
            ax2.set_xlabel("Angle offset Δθ (µrad)")
            ax2.set_ylabel("Reflectivity I/I₀")
            ax2.set_title("Double-crystal (DCM) reflectivity (Bragg)")
            ax2.grid(True)
            st.pyplot(fig2)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.plot(theta_off_urad, rc_curve, color="#2ca02c")
            ax3.set_xlabel("Angle offset Δθ (µrad)")
            ax3.set_ylabel("Reflectivity I/I₀")
            ax3.set_title(f"Analyzer rocking curve")
            ax3.grid(True)
            st.pyplot(fig3)

        with row2_col2:
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            ax4.plot(theta_off_urad, R_analyzer, label="Analyzer crystal", color="#1f77b4")
            ax4.plot(theta_off_urad, R_dcm,      label="Double-crystal",   color="#ff7f0e")
            ax4.plot(theta_off_urad, rc_curve,   label="Rocking curve",    color="#2ca02c")
            ax4.set_xlabel("Angle offset Δθ (µrad)")
            ax4.set_ylabel("Reflectivity I/I₀")
            ax4.set_title("All profiles")
            ax4.grid(True)
            ax4.legend()
            st.pyplot(fig4)
        row3_col1, row3_col2 = st.columns(2)


# ==========================================
# TAB 6 — MIR2 ANALYSIS (single FOV)
# ==========================================
with tab6:
    st.header("MIR2 Analysis (single FOV, Bragg)")
    st.write(
        "Perform per-pixel MIR2 analysis (refraction, USAXS/dark-field, transmission, radiograph) "
        "for a single field of view using Gaussian fits to rocking curves."
    )

    # -----------------------------
    # session state init
    # -----------------------------
    for key, default in [
        ("mir2_raw_dark", None),
        ("mir2_raw_ref", None),
        ("mir2_raw_obj", None),
        ("mir2_shape", None),
        ("mir2_crop_roi", None),
        ("mir2_calib_box", None),
        ("mir2_result", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # -----------------------------
    # helpers
    # -----------------------------
    def _dark2d_for(raw_dark: np.ndarray | None, target_shape: tuple[int, int], target_len: int, idx: int) -> np.ndarray:
        """
        Return a 2D dark for subtraction.
        If raw_dark is None (no files), return zeros (no dark correction).
        """
        if raw_dark is None:
            return np.zeros(target_shape, dtype=np.float64)
    
        if raw_dark.ndim == 2:
            return raw_dark.astype(np.float64)
    
        if raw_dark.ndim != 3:
            raise ValueError(f"raw_dark must be 2D or 3D, got shape {raw_dark.shape}")
    
        nd = raw_dark.shape[0]
        if nd == target_len and idx < nd:
            return raw_dark[idx].astype(np.float64)
        if nd == 1:
            return raw_dark[0].astype(np.float64)
    
        return np.median(raw_dark, axis=0).astype(np.float64)
    
    def _norm_img(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float64)
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return np.zeros_like(img, dtype=np.float64)
        vmin, vmax = np.percentile(finite, [1, 99])
        if vmax <= vmin:
            vmax = vmin + 1e-12
        img = np.clip(img, vmin, vmax)
        return (img - vmin) / (vmax - vmin)
    # -----------------------------
    # ROW 1: three columns
    # -----------------------------
    c1, c2, c3 = st.columns([0.33, 0.34, 0.33])

    # =========================================================
    # ROW 1 / COL 1: Paths + Analyzer reflection/energy + buttons
    # =========================================================
    with c1:
        st.subheader("Paths and MIR parameters")

        dark_dir = st.text_input(
            "Dark frames folder",
            value=r"C:\path\to\dark",
            help="Folder containing dark frames (one TIFF per rocking angle).",
            key="mir2_dark_dir",
        )
        ref_dir = st.text_input(
            "ref frames folder",
            value=r"C:\path\to\ref",
            help="Folder containing ref / reference frames (no object).",
            key="mir2_ref_dir",
        )
        obj_dir = st.text_input(
            "Object frames folder (one FOV/slice)",
            value=r"C:\path\to\object",
            help="Folder containing object frames for a single field of view.",
            key="mir2_obj_dir",
        )
        out_dir = st.text_input(
            "Output folder (save MIR2 results here)",
            value=r"C:\path\to\results",
            key="mir2_out_dir",
        )
        n_angles = st.number_input(
            "Number of rocking-angle frames (optional)",
            min_value=1,
            max_value=1000,
            value=14,
            step=1,
            key="mir2_n_angles_input",
        )
        st.markdown("---")
        st.markdown("**Ideal rocking curve**")
        rc_source = st.radio("Rocking-curve source",
            ["Si model", "Upload ideal RC (text file)"],
            horizontal=True,
            help="Use Si dynamical model, or upload an ideal analyzer RC for non-Si crystals."
        )
        

        rc_upload = None
        rc_upload_unit = "urad"
        if rc_source.startswith("Upload"):
            rc_upload = st.file_uploader(
                "Upload ideal RC (.txt/.csv/.dat): two columns = angle, intensity",
                type=["txt", "csv", "dat"],
                key="mir2_rc_upload",
            )
            rc_upload_unit = st.selectbox(
                "Angle unit in RC file",
                ["urad", "mrad", "rad", "deg", "arcsec"],
                index=0,
                key="mir2_rc_upload_unit",
            )
            st.caption("Comment lines starting with # are allowed. Delimiters: comma, tab, or spaces.")
        st.markdown("---")
        st.markdown("**Analyzer reflection and energy**")

        h_mir = st.number_input("h", value=2, step=1, key="mir2_h")
        k_mir = st.number_input("k", value=2, step=1, key="mir2_k")
        l_mir = st.number_input("l", value=0, step=1, key="mir2_l")
        E_mir = st.number_input("Photon energy (keV)", value=30.0, step=0.5, key="mir2_E")

        geometry_mir = "bragg"

        st.markdown("---")

        load_preview = st.button("1️⃣ Load stacks and show middle-angle preview", key="mir2_load_button")

        # Run button appears only when prerequisites exist
        can_run = (
            st.session_state["mir2_raw_ref"] is not None
            and st.session_state["mir2_raw_obj"] is not None
            and st.session_state["mir2_crop_roi"] is not None
            and st.session_state["mir2_calib_box"] is not None
        )
        run_mir2 = st.button(
            "2️⃣ Run MIR2 analysis for this FOV",
            key="mir2_run_button",
            disabled=not can_run,
        )

    # =========================================================
    # ROW 1 / COL 2: Preview + Crop ROI
    # =========================================================
    with c2:
        st.subheader("Preview of the data and Crop ROI (beam area)")

        # ---------- Load stacks ----------
        if load_preview:
            try:
                raw_ref = load_stack_from_dir(ref_dir, n_angles=None if n_angles is None else int(n_angles))
                raw_obj = load_stack_from_dir(obj_dir, n_angles=None if n_angles is None else int(n_angles))
                raw_dark = load_stack_from_dir(dark_dir, n_angles=None if n_angles is None else int(n_angles), allow_empty=True)
                
                if raw_dark is None:
                    st.info("No dark files found. You can continue without dark correction.")
                    st.checkbox(
                        "Continue without dark correction",
                        key="mir2_continue_no_dark",
                        value=bool(st.session_state.get("mir2_continue_no_dark", False)),
                    )
                    if not st.session_state["mir2_continue_no_dark"]:
                        st.stop()
                
                #either dark exists OR user accepted no-dark
                st.session_state["mir2_raw_dark"] = raw_dark
                st.session_state["mir2_raw_ref"] = raw_ref
                st.session_state["mir2_raw_obj"] = raw_obj
                st.session_state["mir2_shape"] = raw_ref.shape[1:]

                st.session_state["mir2_crop_roi"] = None
                st.session_state["mir2_calib_box"] = None
                st.session_state["mir2_result"] = None
    
                Ny, Nx = raw_ref.shape[1], raw_ref.shape[2]
                st.success(
                    f"Stacks loaded: N_ref={raw_ref.shape[0]}, N_obj={raw_obj.shape[0]}, Ny={Ny}px, Nx={Nx}px."
                        )
                        
            
            except Exception as e:
                st.error(f"Error loading stacks: {e}")

        # ---------- Show middle-angle previews if stacks exist ----------
        if st.session_state["mir2_raw_ref"] is not None:
            raw_dark = st.session_state["mir2_raw_dark"]
            raw_ref  = st.session_state["mir2_raw_ref"]
            raw_obj  = st.session_state["mir2_raw_obj"]

            Ny, Nx = raw_ref.shape[1], raw_ref.shape[2]
            mid_ref = raw_ref.shape[0] // 2
            mid_obj = raw_obj.shape[0] // 2

            dark2d_ref = _dark2d_for(raw_dark, (Ny, Nx), raw_ref.shape[0], mid_ref)
            dark2d_obj = _dark2d_for(raw_dark, (Ny, Nx), raw_obj.shape[0], mid_obj)
            
            ref_mid = (raw_ref[mid_ref].astype(np.float64) - dark2d_ref).clip(min=1.0)
            obj_mid = (raw_obj[mid_obj].astype(np.float64) - dark2d_obj).clip(min=1.0)

            ref_disp = ref_mid / np.max(ref_mid)
            obj_disp = obj_mid / np.max(obj_mid)
            st.info("⏱️ Runtime note: for a dataset of 175 × 4012 pixels, a full MIR2 run is typically ~10 minutes (machine-dependent).")
            st.write("**Middle-angle ref (dark-subtracted)**")
            fig1, ax1 = plt.subplots(figsize=(7, 2.2))
            ax1.imshow(ref_disp, cmap="gray", origin="upper", aspect="auto")
            ax1.set_xlabel("x (pixels)", fontsize=10)
            ax1.set_ylabel("y (pixels)", fontsize=10)
            ax1.set_xticks(np.arange(0, Nx, 300))
            ax1.set_yticks(np.arange(0, Ny, 50))
            ax1.tick_params(axis="both", labelsize=8)
            st.pyplot(fig1)

            st.write("**Middle-angle object (dark-subtracted)**")
            fig2, ax2 = plt.subplots(figsize=(7, 2.2))
            ax2.imshow(obj_disp, cmap="gray", origin="upper", aspect="auto")
            ax2.set_xlabel("x (pixels)", fontsize=10)
            ax2.set_ylabel("y (pixels)", fontsize=10)
            ax2.set_xticks(np.arange(0, Nx, 300))
            ax2.set_yticks(np.arange(0, Ny, 50))
            ax2.tick_params(axis="both", labelsize=8)
            st.pyplot(fig2)

            st.info("Use the pixel axes to choose your crop ROI and calibration box coordinates.")

            st.markdown("---")
            st.markdown("**Crop ROI (beam area)**")
            st.write("Define the large ROI containing the beam (full-image pixel indices).")
            col_roi1, col_roi2 = st.columns(2)
            with col_roi1:
                y_min = st.number_input("Crop y_min", min_value=0, max_value=Ny - 1, value=0, key="mir2_ymin")
                y_max = st.number_input("Crop y_max (exclusive)", min_value=1, max_value=Ny, value=Ny, key="mir2_ymax")
            with col_roi2:
                x_min = st.number_input("Crop x_min", min_value=0, max_value=Nx - 1, value=0, key="mir2_xmin")
                x_max = st.number_input("Crop x_max (exclusive)", min_value=1, max_value=Nx, value=Nx, key="mir2_xmax")

            if y_max <= y_min or x_max <= x_min:
                st.error("Crop ROI: y_max must be > y_min and x_max must be > x_min.")
                st.session_state["mir2_crop_roi"] = None
            else:
                st.session_state["mir2_crop_roi"] = (slice(int(y_min), int(y_max)), slice(int(x_min), int(x_max)))
                st.success(
                    f"Crop ROI set: y=[{int(y_min)}:{int(y_max)}), x=[{int(x_min)}:{int(x_max)})"
                )
            
    # =========================================================
    # ROW 1 / COL 3: Cropped preview + Calibration box
    # =========================================================
    with c3:
        st.subheader("Cropped preview and Calibration box (cropped coordinates)")

        if (
            st.session_state["mir2_raw_ref"] is not None
            and st.session_state["mir2_crop_roi"] is not None
        ):
            raw_dark = st.session_state["mir2_raw_dark"]
            raw_ref  = st.session_state["mir2_raw_ref"]
            raw_obj  = st.session_state["mir2_raw_obj"]
            crop_roi = st.session_state["mir2_crop_roi"]

            mid_ref = raw_ref.shape[0] // 2
            mid_obj = raw_obj.shape[0] // 2

            dark2d_ref = _dark2d_for(raw_dark, (Ny, Nx), raw_ref.shape[0], mid_ref)
            dark2d_obj = _dark2d_for(raw_dark, (Ny, Nx), raw_obj.shape[0], mid_obj)

            dark_crop_ref = dark2d_ref[crop_roi[0], crop_roi[1]]
            dark_crop_obj = dark2d_obj[crop_roi[0], crop_roi[1]]

            ref_crop_mid = (raw_ref[mid_ref, crop_roi[0], crop_roi[1]] - dark_crop_ref).clip(min=1.0)
            obj_crop_mid = (raw_obj[mid_obj, crop_roi[0], crop_roi[1]] - dark_crop_obj).clip(min=1.0)

            ref_c_disp = ref_crop_mid / np.max(ref_crop_mid)
            obj_c_disp = obj_crop_mid / np.max(obj_crop_mid)

            Ny_c, Nx_c = ref_crop_mid.shape

            st.write("**Cropped ref (middle angle)**")
            figc1, axc1 = plt.subplots(figsize=(6, 3))
            axc1.imshow(ref_c_disp, cmap="gray", origin="upper", aspect="auto")
            axc1.set_xlabel("x (pixels, cropped)", fontsize=10)
            axc1.set_ylabel("y (pixels, cropped)", fontsize=10)
            axc1.set_xticks(np.arange(0, Nx_c, max(1, Nx_c // 10)))
            axc1.set_yticks(np.arange(0, Ny_c, max(1, Ny_c // 10)))
            axc1.tick_params(axis="both", labelsize=8)
            st.pyplot(figc1)

            st.write("**Cropped object (middle angle)**")
            figc2, axc2 = plt.subplots(figsize=(6, 3))
            axc2.imshow(obj_c_disp, cmap="gray", origin="upper", aspect="auto")
            axc2.set_xlabel("x (pixels, cropped)", fontsize=10)
            axc2.set_ylabel("y (pixels, cropped)", fontsize=10)
            axc2.set_xticks(np.arange(0, Nx_c, max(1, Nx_c // 10)))
            axc2.set_yticks(np.arange(0, Ny_c, max(1, Ny_c // 10)))
            axc2.tick_params(axis="both", labelsize=8)
            st.pyplot(figc2)

            st.info(f"Cropped image size: Ny_c={Ny_c}px, Nx_c={Nx_c}px.")

            st.markdown("---")
            st.markdown("**Calibration box (cropped coordinates)**")
            st.write("Choose a small box in the cropped image where there is only beam (no sample).")

            col_box1, col_box2 = st.columns(2)
            with col_box1:
                box_y_min_c = st.number_input(
                    "Box y_min (cropped)",
                    min_value=0, max_value=max(0, Ny_c - 1),
                    value=max(0, Ny_c // 2 - 10),
                    key="mir2_box_ymin_c",
                )
                box_y_max_c = st.number_input(
                    "Box y_max (exclusive, cropped)",
                    min_value=1, max_value=max(1, Ny_c),
                    value=min(Ny_c, Ny_c // 2 + 10),
                    key="mir2_box_ymax_c",
                )
            with col_box2:
                box_x_min_c = st.number_input(
                    "Box x_min (cropped)",
                    min_value=0, max_value=max(0, Nx_c - 1),
                    value=max(0, Nx_c // 2 - 10),
                    key="mir2_box_xmin_c",
                )
                box_x_max_c = st.number_input(
                    "Box x_max (exclusive, cropped)",
                    min_value=1, max_value=max(1, Nx_c),
                    value=min(Nx_c, Nx_c // 2 + 10),
                    key="mir2_box_xmax_c",
                )

            if box_y_max_c <= box_y_min_c or box_x_max_c <= box_x_min_c:
                st.error("Calibration box: y_max must be > y_min and x_max must be > x_min.")
                st.session_state["mir2_calib_box"] = None
            else:
                st.session_state["mir2_calib_box"] = (
                    slice(int(box_y_min_c), int(box_y_max_c)),
                    slice(int(box_x_min_c), int(box_x_max_c)),
                )
                st.success(
                    f"Calibration box set (cropped): y=[{int(box_y_min_c)}:{int(box_y_max_c)}), "
                    f"x=[{int(box_x_min_c)}:{int(box_x_max_c)})"
                )

        else:
            st.info("Load stacks and set Crop ROI to enable cropped preview and calibration box.")

    # -----------------------------
    # RUN MIR2 (triggered from COL 1 button)
    # store result in session_state
    # -----------------------------
    if run_mir2:
        try:
            # --- optional uploaded ideal RC for non-Si crystals ---
            rc_upload_bytes = None
            rc_upload_unit_used = None
            if rc_source.startswith("Upload"):
                if rc_upload is None:
                    st.error("Please upload an ideal rocking-curve text file, or switch RC source back to Si model.")
                    st.stop()
                rc_upload_bytes = rc_upload.getvalue()
                rc_upload_unit_used = rc_upload_unit
            raw_dark = st.session_state["mir2_raw_dark"]
            raw_ref  = st.session_state["mir2_raw_ref"]
            raw_obj  = st.session_state["mir2_raw_obj"]
            crop_roi = st.session_state["mir2_crop_roi"]
            calib_box = st.session_state["mir2_calib_box"]

            start_t = time.perf_counter()
            with st.spinner("Running MIR2 analysis..."):
                result = mir2_bragg_single_slice(
                    raw_dark=raw_dark,
                    raw_ref=raw_ref,
                    raw_obj=raw_obj,
                    crop_roi=crop_roi,
                    calib_box=calib_box,
                    h=int(h_mir),
                    k=int(k_mir),
                    l=int(l_mir),
                    energy_keV=float(E_mir),
                    rc_source=rc_source,
                    rc_upload_bytes=rc_upload_bytes,
                    rc_upload_unit=rc_upload_unit_used,
                    geometry=geometry_mir,
                    thickness_mm=float(20.0),
                )
                st.session_state["mir2_result"] = result

                prefix = f"Si{int(h_mir)}{int(k_mir)}{int(l_mir)}_{float(E_mir):.2f}keV"
                save_mir2_results_tiff(result, out_dir=out_dir, prefix=prefix)

            elapsed = time.perf_counter() - start_t
            rt_text = f"{elapsed:.1f} s" if elapsed < 60 else f"{elapsed/60.0:.2f} min"
            st.success(f"MIR2 analysis finished for this FOV in {rt_text}. Saved TIFFs to: {out_dir}")

        except Exception as e:
            st.error(f"Error in MIR2 analysis: {e}")

    # -----------------------------
    # ROW 2: results layout
    #   col 1: rocking curves
    #   col 2-3: contrast images
    # -----------------------------
    st.markdown("---")
    r2c1, r2c23 = st.columns([0.33, 0.67])

    with r2c1:
        st.subheader("Rocking curves in calibration box")
        result = st.session_state["mir2_result"]
        if result is None:
            st.info("Run MIR2 to display rocking curves.")
        else:
            Fbox = np.asarray(result["ref_box_sum"], dtype=np.float64)
            Ibox = np.asarray(result["obj_box_sum"], dtype=np.float64)
            fangl = np.asarray(result["ref_angl"], dtype=np.float64)
            iangl = np.asarray(result["obj_angl"], dtype=np.float64)

            nrmFbox = (Fbox - np.min(Fbox)) / (np.max(Fbox) - np.min(Fbox) + 1e-12)
            nrmIbox = (Ibox - np.min(Ibox)) / (np.max(Ibox) - np.min(Ibox) + 1e-12)

            fig_rc, ax_rc = plt.subplots(figsize=(6.2, 4.0))
            ax_rc.scatter(fangl, nrmFbox, s=18, label="RC for ref data")
            ax_rc.scatter(iangl, nrmIbox, s=18, label="RC for object data")
            ax_rc.set_xlabel("Angle (µrad)")
            ax_rc.set_ylabel("Normalized intensity")
            ax_rc.grid(True)
            ax_rc.legend()
            st.pyplot(fig_rc)

            # Optional: save + download
            fig_rc.savefig(str(Path(out_dir) / "Rocking_Curve.png"), dpi=900, bbox_inches="tight")
            buf = io.BytesIO()
            fig_rc.savefig(buf, format="png", dpi=900, bbox_inches="tight")
           
    with r2c23:
        st.subheader("MIR2 output maps (cropped FOV)")
        result = st.session_state["mir2_result"]
        if result is None:
            st.info("Run MIR2 to display output maps.")
        else:
            ref = result["refraction"]
            wid_abs = result["USAXS"]
            transmission = result["transmission"]
            radio = result["radiograph"]
    
            # Row 1 (2 images)
            c11, c12 = st.columns(2)
            with c11:
                st.write("**refraction**")
                st.image(_norm_img(ref), width="stretch", clamp=True)
            with c12:
                st.write("**USAXS**")
                st.image(_norm_img(wid_abs), width="stretch", clamp=True)
    
            # Row 2 (2 images)
            c21, c22 = st.columns(2)
            with c21:
                st.write("**transmission**")
                st.image(_norm_img(transmission), width="stretch", clamp=True)
            with c22:
                st.write("**radiograph**")
                st.image(_norm_img(radio), width="stretch", clamp=True)
        st.info("Result previews are scaled for visualization; exported TIFFs preserve raw values.")


# ==========================================
# TAB 7 — ABOUT & Cite
# ==========================================
with tab7:
    st.header("About & Citation")
    st.subheader("About")
    st.markdown(
        """
        **MIR2-Toolkit** is a Python-based graphical toolkit for analyzer-based multiple-image radiography (MIR2) data analysis.

        The program is designed to support the complete MIR2 workflow, including:
        - loading dark, reference, and object TIFF image stacks,
        - dark-field correction and ROI selection,
        - theoretical rocking-curve generation for silicon crystals,
        - optional use of externally uploaded rocking curves,
        - angular calibration of reference and object datasets,
        - pixel-wise Gaussian fitting of angular intensity profiles,
        - retrieval of absorption, refraction, and ultra-small-angle X-ray scattering (USAXS) contrast images,
        - interactive visualization of intermediate and final results.

        The toolkit was developed to provide a practical and accessible environment for MIR2 processing through an interactive web interface built with **Streamlit**.

        **Main author:** Farangis Foroughi  
        **Co-author:** David Krapohl  

        **Intended application:**  
        Analyzer-based X-ray phase-contrast imaging and MIR/MIR2 data analysis for research and educational use.

        **Version:** 1.0.0  
        """
    )
    st.subheader("Citation")
    st.markdown(
        """
        If you use this toolkit in your research, please cite:
        
        https://iopscience.iop.org/article/10.1088/1361-6560/ae22ba
        
        **Foroughi, F., et al.**  
        *A Gaussian fitting-based analysis method for multiple image radiography with integrated angular calibration, MIR2.*  
        **Physics in Medicine & Biology** 70(23), 235032 (2025).

        DOI: 10.1088/1361-6560/ae22ba
        
       
        """
    )

