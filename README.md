# Mechanical Metamaterial project

This repository contains the scripts to simulate and process the experimental image data of the mechanical compression of 3D microstructures.

## FEM ## 

The python script "FEM_linear_elasticity.py" simulates the compression of a (micro)structure with the Finite Elements open software Fenics, assuming linear elasticity and specified displacement. The following packages are required:

- Numpy
- Fenics (dolfin) and ufl
- Meshio

The program takes as input a meshed structure in .msh format. It first solves for the displacement field, which is specified at the minimum and maximum z values via Dirichlet boundary conditions, and then computes the resulting strain field. The script loops over several dirichlet boundary conditions in order to calculate several compression steps. The code may be readily edited to simulate different material properties and compression steps, as well as to appropiately tunning the conversion between real and model length units, which is currently set to 2/9.63um.

## Image Registration ##

The reconstruction of the strain and displacement field from experimental data of the microstrucure is done with the open software elastix, which must be previously installed (see https://elastix.lumc.nl/). The software requires a set of images of the undeformed structure, or mask, and a set of images of the deformed sample for each compression step. Each set of images must be assembled into a .tif image stack.

Elastix is run with the bash script "runFFD1223_with_cp_mask.sh". The reconstruction from the two aforementioned stacks is performed according to the options specified in "ffdParameters.txt". Currently the bash script loops over several compression steps. The results of the reconstruction are then stored as nrrd files in the directories "outDir" and "dField".

The resulting .nrrd files then are postprocessed by the python script "plot_sections_p33.py", which requires the following scripts:

- Numpy
- nrrd
- Scikit-image
- Enum
- Matplotlib
- DataClasses
- tomli

The python takes as input the toml file "evalconfig.toml" that contains the path to the nrrd files. It then proceeds to calculate and plot based on the options in this same toml file. Both the python file and the .toml file may be edited to produce different projections according to the user's needs. 

