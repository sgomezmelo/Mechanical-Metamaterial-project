# Mechanical Metamaterial project

This repository contains the scripts for Finite Element Method simulations and image data processing of mechanical compression of microstructures, each of which is in a separate subfolder with the relevant code.  

## FEM ##

The mechanical compression of the microstructure is simulated with the aid of the Finite Element open software fenics. To run the script, the following packages are required:

- Numpy
- Fenics (dolfin) and ufl
- Meshio

The program takes as input a meshed structure in .msh format. It then solves the linear elasticity equations with dirichlet boundary conditions, where displacement is specified at the minimum and maximum z values. The compression is simulated for compression steps of 2um and 5um, and assumes a converstion between real units and model units of 2/9.63um. 

## Image Registration ##
