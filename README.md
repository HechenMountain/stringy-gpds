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

This lets you override default model parameters, the input PDF set, lattice data to show in the plot and which moments to interpolate.
An example user_config.py can also be downloaded at [Zenodo](https://doi.org/10.5281/zenodo.15738460).

### Input PDF sets
The input PDFs are read from `pdfs/<pdf_set>.csv` (unpolarized, MSTW schema) and
`pdfs/<pdf_set>_POL.csv` (polarized, AAC schema). Select a set in `user_config.py`:

```python
pdf_set = "JAM22"   # default: JAM22 PDFs fitted with the GUMP program, mu0 = 2 GeV
mu_input = 2        # input scale of the set in GeV
```

Supplied sets:

| set | input scale | comment |
|-----|-------------|---------|
| `JAM22` | 2 GeV | default, strange quark resolved (`s`, `sbar` fitted separately) |
| `GUMP` | 2 GeV | no strange in the data, sea completed by the kappa-prescription `s=sbar=kappa / 2 (ubar + dbar)`  |
| `MSTW_original` + `AAC_original` | 1 GeV | the original MSTW/AAC parameters (set `pol_pdf_set = "AAC_original"` and `mu_input = 1`) |

## 🔁 Updating the input PDFs
`fit.py` fits a new PDF data set into the parametrization the model requires. The data
format is the one produced by the GUMP program: `x, t, mu, f, delta f, GPD type, flavor`
with GPD type 0 (unpolarized) and 2 (polarized), quarks tabulated for both signs of x
(`f(-x) = -qbar(x)` unpolarized, `f(-x) = +Delta qbar(x)` polarized), and `x*f_g(x)` for the gluon.

Because the moments are built on the input PDFs, a new set invalidates the downstream
model parameters and tables. From the repo's `scripts/` folder:

```bash
# 1. fit the data and activate the set (writes pdfs/<set>.csv + pdfs/<set>_POL.csv,
#    inherits alpha_S(Q0^2) from --base-pdf-set, default: whatever is currently active)
python scripts/fit_input_pdfs.py --pdf-set JAM22 --data PDFdata_JAM22.csv

# 2. Regge slopes / normalizations vs the form-factor / lattice dipoles for the active
#    set -> fit_output/slopes_JAM22.json (singlet A normalizations are held at 1: the
#    forward limit is the input-PDF prediction and A_q(0) + A_g(0) = 1 by the momentum
#    sum rule)
python scripts/refit_slopes.py --pdf-set JAM22

# 3. copy the printed regge_slopes / moment_normalizations into user_config.py, then
#    regenerate the standard-kinematics interpolation tables
python scripts/regen_tables.py

# 4. sanity checks: F1 isovector at t=0 ~ 1, A_q(0)+A_g(0) ~ 1, forward-limit MB
#    closure against the input PDFs
python scripts/validate.py --pdf-set JAM22

# optional: fit-quality (pull) plots
python scripts/plot_pdf_fits.py --pdf-set JAM22 --data PDFdata_JAM22.csv
```

Notes on the fit:
- `A_u`, `A_d`, `A_g` and `x_0` are not fitted but fixed by the number and momentum sum
  rules, so `int uv = 2`, `int dv = 1`, `int (s - sbar) = 0` and `int x [uv+dv+S+g] = 1`
  hold exactly for the written parameters.
- If the data set has no strange entries, `fit_input_pdfs.py` completes the strange sea
  via `s = sbar = kappa/2 (ubar + dbar)` (`kappa` default 0.5, mirrored in the polarized case).
- The functional forms are dictated by the analytic Regge integrals in `regge.py`; a set
  fitted to a *different* parametrization cannot be used.

Without `scripts/` (e.g. a plain `pip install`), the same steps are the Python API directly:

```python
from stringy_gpds.fit import (fit_input_pdfs, fit_polarized_input_pdfs,
                              fit_non_singlet_slopes, fit_singlet_slopes_A,
                              fit_singlet_slopes_Atilde)
import stringy_gpds.config as cfg

fit_input_pdfs("PDFdata_JAM22.csv", pdf_set="JAM22")
fit_polarized_input_pdfs("PDFdata_JAM22.csv", pdf_set="JAM22")
# point user_config.py at pdf_set = "JAM22" and restart the kernel, then:
cfg.memory.clear()   # the cached moments belong to the old PDFs
fit_non_singlet_slopes(evolution_order="nlo")
fit_singlet_slopes_A(evolution_order="nlo", fix_norm_A=True)
fit_singlet_slopes_Atilde(evolution_order="nlo")
# copy the printed values into regge_slopes / moment_normalizations in user_config.py,
# then regenerate the moment tables (interpolate_moments = False, generate_moment_table, ...)
```

## 🧮 Table-generation scripts (`scripts/`)
For batch CSV/plot output over many kinematics at once (rather than one `evolve_conformal_moment`/
`mellin_barnes_gpd` call at a time), `scripts/` has a four-stage pipeline, driven by the same
CLI flags on every stage:

| flag | meaning |
|------|---------|
| `--pdf-set` | input PDF set (default `JAM22`) |
| `--label` | GPD label `H`/`E`/`Htilde`/`Etilde` (default `H`), mapped to a moment label via `GPD_LABEL_MAP` |
| `--particles` | `quark`/`gluon` (default both; `gluon` + non-singlet is skipped, gluon is singlet-only) |
| `--moments` | `non_singlet_isovector`/`non_singlet_isoscalar`/`singlet` (default all three) |
| `--order` | `lo`/`nlo` (default `nlo`) |
| `--errors` | `central`/`plus`/`minus` (default all three) |
| `--eta` | skewness values, e.g. `--eta 0 0.1 0.2` |
| `--mu2` | output scales in GeV², e.g. `--mu2 2 10` |
| `--t-min` or `--t` | either the minimal-\|t\| kinematics at Δ⊥²=0 (`t_min(eta) = -4 eta^2 m_N^2/(1-eta^2)`), or literal `t` values, one per `--eta` |

```bash
# whole pipeline in one call
python scripts/run_gpd_pipeline.py --pdf-set JAM22 --label H \
    --particles quark gluon --moments singlet non_singlet_isovector \
    --eta 0 0.1 0.2 0.3 --mu2 2 10 --t-min

# or checkpoint between stages, e.g. review the cheap moment tables before the
# heavy GPD reconstruction:
python scripts/run_gpd_pipeline.py --stages moments --eta 0 0.1 --mu2 2 10 --t-min
python scripts/run_gpd_pipeline.py --stages interp gpds combine --eta 0 0.1 --mu2 2 10 --t-min
```

Each stage also runs standalone, taking the same flags:
- `generate_moment_tables.py` -- conformal moment tables `F_j` at integer `j` (`--j-max`,
  `--j-min-singlet`, `--j-min-nonsinglet`) -> `data/moments/<SET>/`, `plots/moments/<SET>/`.
- `generate_interp_tables.py` -- moment interpolation tables for non-integer `j`, needed by
  the next stage -> `~/stringy-gpds/data/InterpolationTables/`.
- `generate_gpd_tables.py` -- x-space GPD reconstruction (`--n-points-singlet`,
  `--n-points-nonsinglet` for the x-grid density) -> `data/gpds/<SET>/`, `plots/gpds/<SET>/`.
- `combine_gpd_tables.py` -- reformats the per-eta GPD CSVs into one wide table per
  (particle, moment_type, mu²) with every eta side by side -> `data/gpds/<SET>/combined/`.

If `fit_output/slopes_<pdf-set>.json` exists (written by `refit_slopes.py`) its
`regge_slopes`/`moment_normalizations` override `config.py`'s defaults for that set;
otherwise the set is assumed to already be configured there (true for `JAM22`).

`csv_to_mathematica.py` converts any of these CSVs to a Mathematica-readable `.m` table;
see its own `--help`.

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