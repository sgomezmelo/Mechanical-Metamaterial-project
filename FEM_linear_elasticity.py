from fenics import *
import dolfin
import numpy as np
from ufl import nabla_div, VectorElement, FiniteElement, MixedElement, split
import math 
import meshio
import sys
import os

# Code to solve for a certain geometry in mechanical equilibrium assuming linear elasticity and Dirichlet BC

#Import .msh file of undeformed configuration which is requested as input from user
mesh_name = input("Enter name of .msh file: ")
mesh = dolfin.cpp.mesh.Mesh()
mvc_subdomain = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
mvc_boundaries = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile(MPI.comm_world, mesh_name+".xdmf") as xdmf_infile:
    xdmf_infile.read(mesh)
    xdmf_infile.read(mvc_subdomain, "")

domains = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc_subdomain)
dx = Measure('dx', domain=mesh, subdomain_data=domains)

nu = 0.41 #Material Poisson Ratio
E = 2.9e+9 #Material Young Modulus in Pa
rho = 1100.0 #Material Density in kg/m3
g = 9.81 # Gravity acc in m/s^2
lamb = E*nu/((1+nu)*(1.0-2.0*nu))
mu_la = 1.0/(2.0*nu) - 1.0 #Lame Coefficient Ratio mu/lambda
f = Constant((0.,0.,-rho*g/lamb)) #Body Weight per unit volume
conv_factor = 2.0/9.63 #Conversion factor to model units

#Linear strain and stress tensors
def epsilon(u):
    e = 0.5*(nabla_grad(u) + nabla_grad(u).T) 
    return e 

def stress(u):
    I = Identity(3)
    s = 0.5*inner(I,epsilon(u))*I+2.0*mu_la*epsilon(u)
    return s
    
W = VectorElement("CG", mesh.ufl_cell(), 1) 
Vsig = TensorFunctionSpace(mesh, "CG", degree=1)
V = FunctionSpace(mesh,W)
u_t = TrialFunction(V)
v = TestFunction(V)
d = u_t.geometric_dimension()  
 
class bttm(SubDomain):
    def inside(self,x,on_boundary):
        tol = 5e-2
        zmin = mesh.coordinates()[:, 2].min() #Z coordinate of the bottom plane
        return on_boundary and near(x[2],zmin,tol)
    
class top(SubDomain):
    def inside(self,x,on_boundary):
        tol = 5e-2
        zmax = mesh.coordinates()[:, 2].max() #Z coordinate of the top plane
        return on_boundary and near(x[2],zmax,tol)

x_min = mesh.coordinates()[:, 0].min()
x_max = mesh.coordinates()[:, 0].max()
x_center = (x_min+x_max)/2.0

boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim()-1)

bttm().mark(boundaries, 1) 
top().mark(boundaries, 2) 

dz_top = 0.0 #Prescribed displacement at the top of Anvil
bc2 = DirichletBC(V, Constant((0, 0, dz_top)), boundaries, 1)

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

#Prescribed displacement at the compressing anvil
dz_b = -np.asarray([2.0, 5.0])*conv_factor
du_tilted = 0.0 #Tilted anvil simulated (if no tilting then set to 0)
u_r = [Function(V, name = "Displacement"+str(int(dz_b[i]))) for i in range(len(dz_b))]
eps = [Function(Vsig, name="Strain"+str(int(dz_b[i]))) for i in range(len(dz_b))]

for i in range(len(dz_b)):
    u = u_r[i]
    dz_bttm = dz_b[i]
    sig = eps[i]
    du_top_tilt = Expression((0.0,0.0,"dz_bttm+2.0*du_tilted*(x[0]-x_c)/(x_max-x_min)"), dz_bttm = dz_bttm, du_tilted = du_tilted, x_max = x_max, x_min = x_min, x_c = x_center, degree = 1)
    bc1 = DirichletBC(V, du_top_tilt, boundaries, 2)
    a = inner(stress(u_t),nabla_grad(v))*dx
    L = dot(f,v)*dx
    solve(a == L, u, [bc1, bc2], solver_parameters={'linear_solver': 'gmres', "preconditioner": "ilu"})
    print("Solved for displacement u. Projecting strain into L1 elements ")
    strain = epsilon(u)
    sig.assign(project(strain, Vsig,solver_type = 'gmres', preconditioner_type = "ilu"))
    u.vector()[:] =  u.vector()[:]/conv_factor #Convert back to um
    file_results = XDMFFile(mesh_name+"_elasticity_results_step"+str(int(dz_b[i]))+".xdmf")
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(u, 0.)
    file_results.write(sig, 0.)
    
