# Mechanical Metamaterial project

This repository contains the scripts for Finite Element Method simulations and image data processing of mechanical compression of microstructures, used in the publication --.

## FEM ##plot_sections_p33.py 

The python script "FEM_linear_elasticity.py" simulates the compression of a (micro)structure with the Finite Elements open software Fenics, assuming linear elasticity and specified displacement. The following packages are required:

- Numpy
- Fenics (dolfin) and ufl
- Meshio

The program takes as input a meshed structure in .msh format. It first solves for the displacement field, which is specified at the minimum and maximum z values via dirichlet boundary conditions, and then computes the resulting strain field. Currently, the script simulates compression steps of 2um and 5um, and assumes a conversion of 2 model units to 9.63um (conversion factor of 2/9.63). The code may be readily edited to simulate different compression steps, material properties and appropiate conversion factors.

## Image Registration ##

Given a set of images of the microstructure before and after compression, the strain and displacement field may be reconstructed with the open reconstruction software elastix, which must be previously installed (see https://elastix.lumc.nl/). The following steps explain how to do so. The set of images before and after compression must be collected in a tif image stack. 

The elastix is run with the bash script "runFFD1223_with_cp_mask.sh". This script looks for the undeformed image mask stack and the deformed image stack under the names "fname" and "moving", respectively. Both of these are searched in the path "DataPath". The reconstruction program is then executed with the options specified in "ffdParameters.txt", which is found in the path "paramPath", a subdirectory of "basePath". Currently the bash script loops over several compression steps. The result of the reconstruction is then stored in "outDir" and "dField", a subdirectory of the path "storePath". These store the Jacobian of the transformation as a .nrrd file. 

The resulting .nrrd files then are postprocessed by the python script "plot_sections_p33.py", which requires the following scripts:

- Numpy
- nrrd
- Scikit-image
- Enum
- Matplotlib
- DataClasses
- tomli

The python program looks for the elastix results  from the path specified in the "evalconfig.toml". It then proceeds to calculate and plot based on the options in this same toml file. Both the python file and the .toml file may be edited to produce different projections according to the user's needs. 

