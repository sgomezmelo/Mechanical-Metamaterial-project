# Mechanical Metamaterial project

This repository contains the scripts for simulation and data processing of mechanical compression of microstructures, relevant to the publication ---. It is divided into two subdirectories: the FEM, which simulates the compression with the aid of the Finite Elements Method assuming linear elasticity regime, and the Image registration, which reconstructs the displacement and strain fields from experimentally obtained images.  

## FEM ##

To simulate the mechanical compression, we solve the equations of linear elasticity using the Finite Element open software Fenics. The following packages are needed

- Numpy
- Fenics (dolfin) and ufl
- Meshio

The script takes as input the compressed microstructure in .msh format, and simulates compressions of 2um and 5um. 
