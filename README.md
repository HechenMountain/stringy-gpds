# stringy_gpds

**A string-based parametrization of Generalized Parton Distribution functions (GPDs)** 
## About
A Python package that utilizes a string-based parametrization of quark and gluon conformal moments to reconstruct Generalized Parton Distribution functions (GPDs). It leverages an expansion in conformal partial waves and conformal moments to compute GPDs via Mellin-Barnes integrals, accessible over the whole physical region in parton x, skewness eta and Mandelstam t.

## 📦 Features

- Up to next-to-leading-order evolution of conformal moments for quarks and gluons. Including non-diagonal evolution for complex conformal spin.
- Reconstruction of evolved GPDs via resummation of the conformal moment expansion through a complex Mellin-Barnes integral.
- Momentum space and impact parameter space representation.
- Fast reconstruction due to caching, interpolation and dipole fits for impact parameter representation.
- Spin and orbital angular momentum decomposition, spin-orbit correlation.
- Dedicated plotting functions.
- Automatic data handling to compare with available data. Easily extended by user to accomodate new data.
- Currently supports (un)polarized H and E GPDs and their moments.
- Different choice of input PDF sets.
- User-friendly modification of model parameters.

## 🛠 Installation

```bash
pip install git+https://github.com/HechenMountain/stringy-gpds.git
```

## ⚙️ Configuration
This code relies on tables that are used for interpolation which can be found at [Zenodo](https://doi.org/10.5281/zenodo.15738460).
The folder contained the tables need to be placed at:
- Linux/macOS: `~/stringy-gpds/`
- Windows: `C:\Users\your-username\stringy-gpds\`

Optional user-specific settings can be defined in a `user_configy.py` file placed in the same folder.

This lets you override default model parameters, lattice data to show in the plot and which moments to interpolate.
An example user_config.py can also be downloaded at [Zenodo](https://doi.org/10.5281/zenodo.15738460).

## 🚀 Example Usage
On first execution, the program generates interpolators and computes error estimates for the GPDs which are being cached on the filesystem.
You can specify which functions are interpolated and which error metrics are calculated in user_config.py (see [Zenodo](https://doi.org/10.5281/zenodo.15738460)).
Optionally, the non-diagonal evolution can be kept for the evaluation of the Mellin-Barnes integral.

```python
from stringy_gpds import evolve_conformal_moment, mellin_barnes_gpd
# To obtain the evolved moments use e.g.
evolve_conformal_moment(j=2,eta=0.33,t=-0.69,mu=2,particle="gluon",moment_type="singlet",moment_label="Atilde",evolution_order="nlo")

# Particles are "quark" and "gluon" with moments "non_singlet_isovector", "non_singlet_isoscalar" and "singlet". 
# The moment_label corresponds to the standard nomenclature used in the literature, but with the D-term implicit.
# I.e. A(tilde) for moments of H(tilde) GPDs and B for moments of E GPD.
# The Regge slopes and normalizations are defined in user_config.py. If a different PDF set is used, the corresponding functions to fit
# to form factors are fit_non_singlet_slopes etc.

# To obtain the GPD at a particular value of parton x use
mellin_barnes_gpd(x=.2,eta=0.33,t=-0.69,mu=2,particle="gluon",moment_type="singlet",moment_label="Atilde",evolution_order="nlo")

# For the GPD reconstruction over the whole x region it is recommended to interpolate the moments
# for complex values of conformal spin-j. This is done by setting interpolate_moments = False in user_config.py
# and use generate_moment_table for the desired kinematics eta, t and resolution scale mu. 
# Afterwards, the kernel should be restarted, interpolate_moments = True
# and the desired moments to interpolate should be defined in user_config.py using e.g.
eta_array = [0.0, 0.33, 0.1]
t_array = [-0.69, -0.69, -0.23]
mu_array = [2, 2, 2]

particles = ["gluon"]
moments = ["singlet"]
labels = ["Atilde"]
orders = ["nlo"]
errors = ["central", "plus", "minus"]

# To generate the data over the whole region in parton x use
import stringy_gpds.config as cfg
from stringy_gpds import plot_gpds
colors = ["purple","green","red"]
plot_gpds(eta_array,t_array,mu_array,colors,particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",evolution_order="nlo",error_bars=True, read_from_file= False,write_to_file=True, y_0=0, y_1=2.5,plot_legend=True)

# To plot the lattice data as well use plot_gpd_data. 
# This will automatically use the kinematics defined in gpd_publication_mapping in user_config.py
# for a given set of (particle,gpd_type,gpd_label)

# For fast numerical Fourier transforms it is recommended to generate dipole fits using e.g.
from stringy_gpds import dipole_fit_moment
dipole_fit_moment(n=1,eta=0,mu=2,particle="quark",moment_type="non_singlet_isovector",moment_label="Atilde")
# This generates a csv containing the dipole parameters. If the function is called for various combinations
# of (n,particle,moment_type,moment_label) the parameters are appended/updated in the same csv.

# and then
from stringy_gpds import plot_fourier_transform_singlet_helicity
plot_fourier_transform_singlet_helicity(n=0,mu=2,particle="quark",vmin=0,vmax=0.7,ymin=0,ymax=1,read_from_file=False,write_to_file=True)
# This will additionaly save the data as csv to the filesystem such that it can be read from the filesystem using 
# read_from_file=True and write_to_file=False

# Additional examples as well as the code used to 
# generate the plots in the publication can be found in 
# StringBasedGPD.ipynb
```

## 💬 Additional Comments
Carefully read user_config.py and or config.py (within the package folder), it should be self-explanatory. 
All functions in the source code are equipped with docstrings.
If there is still something unclear after reading the docstrings and config.py, do not hesitate to contact me!

All dimensionful quantities are given in units of GeV with conversion to fm only for the plots.

For the GPD reconstruction, the non-diagonal part of the evolution equations can be discarded by using nd_evolved_complex_moment = False (recommended) 
since the contribution is < 5%.

For the data handling include ArXiv IDs in publication_mapping for the moments and gpd_publication_mapping for GPDs in user_config.py.

Initial tables for interpolation are supplied for harmonic numbers, anomalous dimensions and some moments.

## 📁 Data Access

The full dataset (PDFs, CSV tables for interpolation, extracted lattice data and data used for plot generation) 
is available at [Zenodo](https://doi.org/10.5281/zenodo.15738460). The contents of the data folder need to be placed in:
- Linux/macOS: `~/stringy_gpds/data`
- Windows: `C:\Users\your-username\stringy_gpds\data`
And the contents of the pdfs folder need to be placed in:
- Linux/macOS: `~/stringy_gpds/pdfs`
- Windows: `C:\Users\your-username\stringy_gpds\pdfs`


## 📊 Lattice data
The `.csv` files under [Zenodo](https://doi.org/10.5281/zenodo.15738460) containing lattice data
were manually extracted from published results in the following works:
- JHEP 01 (2025) 146 • e-Print: 2410.03539 [hep-lat]
- Phys.Rev.D 110 (2024) 3, 3 • e-Print: 2312.10829 [hep-lat]
- Phys.Rev.Lett. 132 (2024) 25, 251904 • e-Print: 2310.08484 [hep-lat]
- Phys.Rev.D 108 (2023) 1, 014507 • e-Print: 2305.11117 [hep-lat]
- Phys.Rev.D 106 (2022) 11, 114512 • e-Print: 2209.05373 [hep-lat]
- Phys.Lett.B 824 (2022) 136821 • e-Print: 2112.07519 [hep-lat]
- Phys.Rev.D 101 (2020) 3, 034519 • e-Print: 1908.10706 [hep-lat]
- Phys.Rev.Lett. 125 (2020) 26, 262001 • e-Print: 2008.10573 [hep-lat]
- Phys.Rev.D 77 (2008) 094502 • e-Print: 0705.4295 [hep-lat]

These files are provided **for reproducibility purposes only**. 
The maintainer of this package claims **no ownership** of the original data.
Please cite the original authors if you use this data in scientific work.



## 🐛 Issues & Support

If you encounter any problems, have questions, or want to request a feature, feel free to open an issue on the [GitHub Issue Tracker](https://github.com/HechenMountain/stringy-gpds/issues).

## 📈 Plots 
The plots are automatically saved to the folder:
- Linux/macOS: `~/stringy_gpds/plots`
- Windows: `C:\Users\your-username\stringy_gpds\plots`

It needs to be either created by the user or by running the first cell in the ipynb.

## 📄 License

- **Software** is licensed under the [MIT License](LICENSE).

## 📖 How to Cite

If you use this code or data, please cite:

Hechenberger, F. Mamo, K. A., Zahed, I. (2025). Rapidity-Dependent Spin Decomposition of the Nucleon, 2507.18615