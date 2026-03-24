# MIR2-Web-Toolkit
MIR2 Web Toolkit is a Python-based graphical software tool for analyzer-based multiple-image radiography (MIR2) data analysis. The toolkit provides an interactive Streamlit interface for silicon crystal diffraction calculations, rocking-curve generation, optional external rocking-curve upload, angular calibration, pixel-wise Gaussian fitting of angular intensity profiles, and retrieval of MIR contrast channels, including transmission/radiograph, refraction, and ultra-small-angle X-ray scattering (USAXS). The software is intended for research use in analyzer-based X-ray phase-contrast imaging.

## Main features

- Bragg angle calculation for Si reflections
- Forward susceptibility calculation for silicon
- Structure factor calculation for silicon
- Darwin width calculation
- Reflectivity and rocking-curve simulation
- Optional upload of an external rocking curve
- MIR2 analysis of dark, reference, and object TIFF stacks
- Independent angular calibration of reference and object datasets
- Pixel-wise Gaussian fitting
- Interactive visualization of intermediate and final outputs
- Included sample datasets for testing
  
## Requirements

- Python 3.12 or compatible
- Windows, macOS, or Linux

**Note:** The current version of the toolkit has been tested with Python 3.12.

## Installation

### 1. Install Python

Download Python from the official Python website:

https://www.python.org/downloads/

If you are using Windows, it is recommended to enable **Add Python to PATH** during installation.

### 2. Download or clone the repository
