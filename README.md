# MIR2-Web-Toolkit
MIR2 Web Toolkit is a Python-based graphical software tool for analyzer-based multiple-image radiography (MIR2) data analysis. The toolkit provides an interactive Streamlit interface for silicon crystal diffraction calculations, rocking-curve generation, optional external rocking-curve upload, angular calibration, pixel-wise Gaussian fitting of angular intensity profiles, and retrieval of MIR contrast channels, including transmission/radiograph, refraction, and ultra-small-angle X-ray scattering (USAXS). The software is intended for research use in analyzer-based X-ray phase-contrast imaging.

## Repository

GitHub repository:  
https://github.com/farangisforoughi/MIR2-Web-Toolkit/tree/main

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
#### Option 1: Clone with Git

```bash
git clone https://github.com/farangisforoughi/MIR2-Web-Toolkit.git
cd MIR2-Web-Toolkit
```

#### Option 2: Download as ZIP

Open the GitHub repository page:

https://github.com/farangisforoughi/MIR2-Web-Toolkit/tree/main

Then click **Code** -> **Download ZIP**, extract the folder, and open the extracted folder in a terminal.

### 3. Create a virtual environment

It is recommended to use a virtual environment to avoid package conflicts.
#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the required packages

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

Start the Streamlit app with:

```bash
python -m streamlit run app.py
```

A local browser window should open automatically. If it does not, copy the local URL shown in the terminal and paste it into your browser.

## Program overview

The toolkit contains several tabs. The first tabs provide theoretical calculations related to silicon crystal diffraction, while the MIR2 Analysis tab performs the image-based MIR2 workflow.

- **Tab 1 – Bragg Angle Calculator**  

- **Tab 2 – Forward Susceptibility**  
 
- **Tab 3 – Structure Factor**  
 
- **Tab 4 – Darwin Width**  
  
- **Tab 5 – Reflectivity & Rocking Curve**  
 
- **Tab 6 – MIR2 Analysis**  

- **Tab 7 – About & Citation**  
  

## Typical workflow

1. Launch the app with `python -m streamlit run app.py`
2. Open the MIR2 Analysis tab (Review the silicon diffraction tabs if needed, but the MIR2 analysis tab works independently)
3. Load the dark, reference, and object image stacks
4. 5. Choose the rocking-curve source: internal silicon-model rocking curve or uploaded external rocking curve
5. Set experiment and fitting parameters
6. Define the crop ROI and calibration box
7. Run the MIR2 analysis
8. Inspect and save the results
## Step-by-step MIR2 analysis

### Step 1. Launch the app

Run:

```bash
python -m streamlit run app.py
```

### Step 2. Open the MIR2 Analysis tab

Navigate to the tab dedicated to MIR2 image processing.

### Step 3. Load the input data

Load or select the three required TIFF image stacks:

- dark (optional)
- reference
- object

The software expects TIFF images arranged in separate folders for each dataset.

### Step 4. Select the rocking-curve source

Choose one of the following options:

**Option A: Use the internal silicon model**  
Use the built-in silicon diffraction model to generate the rocking curve. This is suitable when the analyzer crystal is silicon and the experimental configuration matches the model assumptions.

**Option B: Upload an external rocking curve**  
Upload a user-provided rocking curve if:
- a non-silicon analyzer is used
- an experimentally measured rocking curve is preferred
- a custom analyzer response is required

### Step 5. Set experiment parameters

Specify the required parameters, depending on the selected mode. These include:

- crystal reflection `(hkl)`
- X-ray energy

### Step 6. Define the crop ROI

Select the region of interest (ROI) for the area you want to analyze. This reduces the processed image size and focuses the analysis on the relevant area.

### Step 7. Define the calibration box

Select a suitable calibration region used for angular calibration.

A good calibration box should ideally be placed in:
- air, or
- a uniformly low-scattering, unstructured region

A structured region should be avoided, because it can distort the measured scattering width and reduce calibration quality.

### Step 8. Run the analysis

The software performs the MIR2 workflow, including:

- dark correction
- independent processing of reference and object datasets
- angular calibration
- Gaussian fitting of angular intensity profiles
- retrieval of MIR contrast images

### Step 9. Inspect the results

The toolkit displays intermediate and final outputs such as:

- rocking-curve plot
- transmission and radiograph
- refraction image
- USAXS image

## Example data

Example datasets are included in the repository to help users test the software before analyzing their own experimental data. These sample datasets can be used to verify that the installation works, explore the interface, and understand the expected folder structure.

## Citation

If you use this toolkit in your research, please cite:

**Foroughi, F., et al.**  
*A Gaussian fitting-based analysis method for multiple image radiography with integrated angular calibration, MIR2.*  
**Physics in Medicine & Biology** 70(23), 235032 (2025).  
DOI: `10.1088/1361-6560/ae22ba`

## Authors

**Software main author:** Farangis Foroughi  
**Software co-author:** David Krapohl
