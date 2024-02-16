# Mechanical Metamaterial project

This repository contains the scripts for Finite Element Method simulations and image data processing of mechanical compression of microstructures, used in the publication --.

## FEM ##

The mechanical compression of the microstructure is simulated with the aid of the Finite Element open software fenics. To run the script, the following packages are required:

- Numpy
- Fenics (dolfin) and ufl
- Meshio

The program takes as input a meshed structure in .msh format. It then solves the linear elasticity equations with dirichlet boundary conditions, where displacement is specified at the minimum and maximum z values. The compression is simulated for compression steps of 2um and 5um, and assumes a conversion between real units and model units of 2/9.63um. The code may be readily edited to simulate different compression steps and appropiate conversion factors.

## Image Registration ##

Given a set of images of the microstructure before and after compression, the strain and displacement field may be reconstructed with the open software elastix, which must be previously installed (see https://elastix.lumc.nl/). 
