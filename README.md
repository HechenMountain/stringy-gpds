# stringy_gpds

**String-based parametrization of Generalized Parton Distribution functions (GPDs)** — About
A Python package that utilizes a string-based parametrization of quark and gluon conformal moments to reconstruct Generalized Parton Distribution functions (GPDs). It leverages an expansion in conformal partial waves and conformal moments to compute GPDs via Mellin-Barnes integrals, accessible over the whole physical region in parton x, skewness eta and Mandelstam t.

## 📦 Features

- Evolution of conformal moments for quarks and gluons.
- Reconstruction of GPDs via Mellin-Barnes techniques.
- Fast reconstruction due to caching and interpolation.
- Spin and orbital angular momentum decomposition, spin-orbit correlation.
- Momentum space and impact parameter space representation.
- Dedicated plotting functions.
- Automatic data handling to compare with data.
- Up to next-to-leading-order evolution including non-diagonal parts in evolutionequation.
- Includes H, Htilde and E GPDs and their moments

## 🛠 Installation

```bash
pip install git+https://github.com/HechenMountain/stringy-gpds.git
```
## 🧾 Example Usage

```python
# To obtain the evolved moments use e.g.
evolve_conformal_moment(j=2,eta=0.33,t=-0.69,mu=2,particle="gluon",moment_type="singlet",moment_label="Atilde",evolution_order="nlo")

# Particles are "quark" and "gluon" with moments "non_singlet_isovector", "non_singlet_isoscalar" and "singlet". The moment_label corresponds to the standard nomenclature used in the literature. I.e. A(tilde) for moments of H(tilde) GPDs and B for moments of E GPD. The Regge slopes and normalizations are defined in config.py

# To obtain the GPD at a particular value of parton x use
mellin_barnes_gpd(x=.2,eta=0.33,t=-0.69,mu=2,particle="gluon",moment_type="singlet",moment_label="Atilde",evolution_order="nlo")

# For the GPD reconstruction over the whole x region it is recommended to interpolate the moments for complex values of conformal spin-j. This is done by setting INTERPOLATE_MOMENTS = False in config.py and use generate_moment_table for the desired kinematics eta, t and resolution scale mu. Afterwards, the kernel should be restarted, INTERPOLATE_MOMENTS = True and the desired moments to interpolate should be defined in config.py using e.g.

ETA_ARRAY = [0,0.33,0.1]
T_ARRAY = [-0.69,-0.69,-0.23]
MU_ARRAY = [2,2,2]

PARTICLES = ["quark","gluon"]
MOMENTS = ["singlet","non_singlet_isoscalar","non_singlet_isovector"]
LABELS = ["A","Atilde"]
ORDERS = ["nlo"]
ERRORS = ["central","plus","minus"]

# To generate the data over the whole region in parton x use
import stringy_gpds.config as cfg
eta_array = cfg.ETA_ARRAY
t_array = cfg.T_ARRAY
mu_array = cfg.MU_ARRAY
colors = ["purple","green","red"]
plot_gpds(eta_array,t_array,mu_array,colors,particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",evolution_order="nlo",error_bars=True, read_from_file= False,write_to_file=True, y_0=0, y_1=2.5,plot_legend=True)

# To plot the lattice data as well use plot_gpd_data. This will automatically use the kinematics defined in GPD_PUBLICATION_MAPPING in config.py for a given set of (particle,gpd_type,gpd_label)
```
## Additional comments
Read config.py, it should be self-explanatory. 

All functions in the source code are equipped with docstrings.

All dimensionful quantities are given in units of GeV with conversion to fm only for the plots.

For the GPD reconstruction, the non-diagonal part of the evolution equations can be discarded by using ND_EVOLVED_COMPLEX_MOMENT = False (recommended) since the contribution is < 5%

For the data handling. Include ArXiv IDs in PUBLICATION_MAPPING for the moments and GPD_PUBLICATION_MAPPING for GPDs.

Initial tables for interpolation are supplied for harmonic numbers, anomalous dimensions and some moments.

## Lattice data is from:
- JHEP 01 (2025) 146 • e-Print: 2410.03539 [hep-lat]
- Phys.Rev.D 110 (2024) 3, 3 • e-Print: 2312.10829 [hep-lat]
- Phys.Rev.Lett. 132 (2024) 25, 251904 • e-Print: 2310.08484 [hep-lat]
- Phys.Rev.D 108 (2023) 1, 014507 • e-Print: 2305.11117 [hep-lat]
- Phys.Lett.B 824 (2022) 136821 • e-Print: 2112.07519 [hep-lat]
- Phys.Rev.D 101 (2020) 3, 034519 • e-Print: 1908.10706 [hep-lat]
- Phys.Rev.Lett. 125 (2020) 26, 262001 • e-Print: 2008.10573 [hep-lat]
- Phys.Rev.D 77 (2008) 094502 • e-Print: 0705.4295 [hep-lat]