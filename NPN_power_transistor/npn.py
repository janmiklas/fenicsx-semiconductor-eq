import gmsh
import numpy as np
#import pyvista
import matplotlib.pyplot as plt
from myplot import myplot_2d, pltsubdomains, myplot2Dmoj, project, myeval
import myplot
import os
from petsc4py import PETSc

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                                 form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities
from dolfinx.mesh import CellType, create_rectangle, locate_entities, locate_entities_boundary, meshtags, refine
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                         dx, grad, inner, div,
                         FiniteElement, VectorElement, MixedElement, TestFunctions, TrialFunctions)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import ufl

from params import *

EPS = np.spacing(1)

############
### MESH ###
############
mesh = create_rectangle(
    MPI.COMM_WORLD,
    points=((0,0), (XLEN, YLEN)),
    n=(200, 200),
    cell_type=CellType.triangle,
)

tag_bc_E = 28
tag_bc_B = 29
tag_bc_C = 30
#tag_bc_medzera = 31

#######################
### Function Spaces ###
#######################

elem_poiss = FiniteElement("CG", mesh.ufl_cell(), 1)
elem_cont = FiniteElement("CG", mesh.ufl_cell(), 1)
elem_mix = MixedElement([elem_poiss, elem_cont, elem_cont])
V = FunctionSpace(mesh, elem_mix)
#
(v0, v1, v2) = TestFunctions(V)
#q = Function(V)
#u, y = split(q)
#alebo? TrialFunctions(

u = Function(V)
#(u0, u1, u2) = (Psi, p, n) = (Psi, p, n) = u.split() ### TOTO by sa mi pozdavalo viac, ale neskonvergoval Newton solver!
(Psi, p, n) =  ufl.split(u) # po tomto v pohode skonvergoval
#u = TrialFunction(V)
#u0 = Psi = u[0]
#u1i = p = u[1]
#u2 = n = u[2]
#(u0, u1, u2) = (Psi, p, n) = TrialFunctions(V)

###########################
### Doping profiles  ###
###########################
print("doping...")

Q = FunctionSpace(mesh, elem_poiss)

def distance(x, Yab=YLEN, Xa=.5, Xb=1.5):
    d1 = (x[0]<Xa) * ((Xa-x[0])**2 + (Yab-x[1])**2)**.5
    d2 = (x[0]>Xb) * ((Xb-x[0])**2 + (Yab-x[1])**2)**.5
    d3 = np.logical_and((x[0]>=Xa), (x[0]<=Xb)) * ((Yab-x[1])**2)**.5
    return d1+d2+d3
#uD.interpolate(distance)
#######################################3
# lubovolna matematicka funkcia f(distance)
def doping_log10(dist, C0, Clen):
    ret = (dist <= Clen) * C0*(np.log10(-dist*(10-1)/Clen +10))**0.5
    return np.nan_to_num(ret, nan=0)
def doping_log10(dist, C0, Clen):
    ret = (dist <= Clen) * C0*(np.log10(-dist*(10-1)/Clen +10))**0.3
    return np.nan_to_num(ret, nan=0)
def doping_constant(dist, C0, Clen):
    ret = (dist <= Clen) * C0
    return np.nan_to_num(ret, nan=0)
def doping_gauss6(dist, C0, Clen):
    #ret = (dist <= Clen) * C0*(np.log10(-dist*(10-1)/Clen +10))**0.3
    ret = np.exp(-dist**6/2/.1**2)
    ret = C0 * np.exp(-(dist/Clen)**6/2/(1)**2)
    return np.nan_to_num(ret, nan=0)
def doping_gauss(dist, C0, Clen, exponent=10):
    return C0 * np.exp(-(dist/Clen)**exponent/2/(1)**2)
def doping_expexp(dist, C0, Clen, exponent=4):
    #return C0 * np.exp(-(np.exp(dist/6)-1)**6)
    return C0 * np.exp(-(np.exp(dist/1.44/Clen)-1)**exponent)
#######################################3
#######################################3

exponent = 6
def dop1(x, Yab=YLEN, Xa=xBE1, Xb=XLEN, C0=Nd1, Clen=ylenE):
    d = distance(x, Yab=Yab, Xa=Xa, Xb=Xb)
    return doping_gauss(d, C0, Clen, exponent=exponent)

def dop2(x, Yab=YLEN, Xa=0, Xb=XLEN, C0=Na2, Clen=ylenB):
    d = distance(x, Yab=Yab, Xa=Xa, Xb=Xb)
    return doping_gauss(d, C0, Clen, exponent=exponent)

def dop4(x, Yab=0, Xa=0, Xb=XLEN, C0=Nd4, Clen=ylenE):
    d = distance(x, Yab=Yab, Xa=Xa, Xb=Xb)
    return doping_gauss(d, C0, Clen, exponent=exponent)

#doping_E = Function(V.sub(0))
#doping_B = Function(V.sub(0))
#doping_C = Function(V.sub(0))
doping_drift = Function(Q)
doping_E = Function(Q)
doping_B = Function(Q)
doping_C = Function(Q)

#doping_drift = Nd3
print("background doping ... drift...")
doping_drift.x.array[:] = Nd3
print("interpolating doping function... E...")
doping_E.interpolate(dop1)
print("interpolating doping function... B...")
doping_B.interpolate(dop2)
print("interpolating doping function... C...")
doping_C.interpolate(dop4)

Nd = doping_E + doping_drift + doping_C
Na = doping_B
N = Nd-Na
N_proj = project(N, Q)

if not os.path.isdir("results"):
    os.system("mkdir results")
print("plot aa...")
myplot2Dmoj(N_proj, savename="doping")


#Ndoping = materialy(Nd1-Na1, Nd2-Na2, Nd3-Na3, Nd4-Na4)

###########
## initial:
###########
print("initial...")

# pre Nd pozostavajuce z jedneho dopingu by project ani nebolo treba, ale pri sucte viacerych ma nenapada lepsi sposob
# mozno scitat vektory doping.x.array a potom zapisat do spolocneho
print("projection Nd...")
ninit = project(Nd, Q)
print("projection Na...")
pinit = project(Na, Q)

u.sub(1).interpolate(pinit)
u.sub(2).interpolate(ninit)
#u.sub(0).interpolate(Psiinit)

#####################
## poriadny sposob 
print("initial...")

##pripadne nie novy space Q, ale:
V1, map1 = V.sub(1).collapse()
V2, map2 = V.sub(2).collapse()
## a potom na zaklade tohto mapovania zapisovat hodnoty priamo do funkcie v mixed space Function(V)
#pinit = Function(V1) # alebo aj Q kludne, ale takto netreba davat pozor
#ninit = Function(V2)
#N_proj = project(N, V1)
pinit_array = np.clip(-N_proj.x.array, 0, np.inf) # np.clip: oreze hodnoty na definovanu min a max hodnotu
ninit_array = np.clip(N_proj.x.array, 0, np.inf)
u.x.array[map1] = pinit_array
u.x.array[map2] = ninit_array
myplot2Dmoj(u.sub(1), savename="pinit", aspect_equal=False)
myplot2Dmoj(u.sub(2), savename="ninit", aspect_equal=False)

###########################
### Boundary Conditions ###
###########################
## z tutorialu https://jsdokken.com/dolfinx-tutorial/chapter3/component_bc.html
#def right(x):
#    return np.logical_and(np.isclose(x[0], L), x[1] < H)
#boundary_facets = locate_entities_boundary(mesh, mesh.topology.dim-1, right)
#boundary_dofs_x = locate_dofs_topological(V.sub(0), mesh.topology.dim-1, boundary_facets)
#bcx = dirichletbc(ScalarType(0), boundary_dofs_x, V.sub(0))

print("dirichlet BC...")

def contact_E(x):
    return np.logical_and(np.isclose(x[1], YLEN), x[0] > xBE1)
    #return np.logical_and(np.isclose(x[1], YLEN), x[0] > xBE1+.02*xlenB)
facets_bc_E = facets = locate_entities_boundary(mesh, mesh.topology.dim-1, contact_E) # alebo rovnako aj locate_entities (to len tie entity vrati v inom poradi). locate_entities ich mimochodom vracia zoradene 
#tag=tag_bc_E
#facets = facets_bc_E = ft.indices[ft.values==tag]
dofs0 = dofs0_bc_E = locate_dofs_topological(V, mesh.topology.dim-1, facets)
dofs1 = dofs1_bc_E = locate_dofs_topological(V.sub(1), mesh.topology.dim-1, facets)
dofs2 = locate_dofs_topological(V.sub(2), mesh.topology.dim-1, facets)
bc_psi_E = dirichletbc(ScalarType(Psi1), dofs0, V.sub(0))
bc_p_E = dirichletbc(ScalarType(0), dofs1, V.sub(1))
bc_n_E = dirichletbc(ScalarType(Nd1+Nd3-Na2), dofs2, V.sub(2))

def contact_B(x):
    return np.logical_and(np.isclose(x[1], YLEN), x[0] < xlenBcontact)
facets_bc_B = facets = locate_entities_boundary(mesh, mesh.topology.dim-1, contact_B)
#tag=tag_bc_B
#facets = facets_bc_B = ft.indices[ft.values==tag]
dofs0 = dofs0_bc_B = locate_dofs_topological(V, mesh.topology.dim-1, facets)
dofs1 = dofs1_bc_B = locate_dofs_topological(V.sub(1), mesh.topology.dim-1, facets)
dofs2 = locate_dofs_topological(V.sub(2), mesh.topology.dim-1, facets)
bc_psi_B = dirichletbc(ScalarType(Psi2), dofs0, V.sub(0))
bc_p_B = dirichletbc(ScalarType(Na2-Nd3), dofs1, V.sub(1))
bc_n_B = dirichletbc(ScalarType(0), dofs2, V.sub(2))

def contact_C(x):
    return np.isclose(x[1], 0)
facets_bc_C = facets = locate_entities_boundary(mesh, mesh.topology.dim-1, contact_C)
#tag=tag_bc_C
#facets = facets_bc_C = ft.indices[ft.values==tag]
dofs0 = dofs0_bc_C = locate_dofs_topological(V, mesh.topology.dim-1, facets)
dofs1 = locate_dofs_topological(V.sub(1), mesh.topology.dim-1, facets)
dofs2 = locate_dofs_topological(V.sub(2), mesh.topology.dim-1, facets)
bc_psi_C = dirichletbc(ScalarType(Psi3), dofs0, V.sub(0))
bc_p_C = dirichletbc(ScalarType(0), dofs1, V.sub(1))
bc_n_C = dirichletbc(ScalarType(Nd4+Nd3), dofs2, V.sub(2))


#bc_n_p = dirichletbc(ScalarType(ni**2/Na1), dofs2, V.sub(2))

# https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
facet_markers_list = [tag_bc_E, tag_bc_B, tag_bc_C]
facet_indices = [facets_bc_E, facets_bc_B, facets_bc_C]

facet_markers = [np.full_like(facet_indices[i], facet_markers_list[i]) for i in range(len(facet_markers_list))]
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tags = meshtags(mesh, mesh.topology.dim-1, facet_indices[sorted_facets], facet_markers[sorted_facets])


ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tags)  
dS = ufl.Measure('dS', domain=mesh, subdomain_data=facet_tags)  
#dxx = ufl.Measure('dx', domain=mesh, subdomain_data = ct)

#ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)  
#dS = ufl.Measure('dS', domain=mesh, subdomain_data=ft)  
#dxx = ufl.Measure('dx', domain=mesh, subdomain_data = ct)


###############
print("variational forms...")

###############
### POISSON ###
###############

a0 = inner(grad(Psi), grad(v0)) * dx - 1/lmda0 * (Nd-Na-n+p + EPS) * v0 * dx 
L0 = 0

nn=ufl.FacetNormal(mesh)
### Neumannova BC
# pre emitor:
J = 0e-10
#g = -(J+grad(p)[0])/p/mob_p
#g = -(Dp*inner(grad(p), nn) - inner(J, nn))/(p*mob_p) # ked J by bol vektor
g = (-Dp*inner(grad(p), nn) - J)/(p*mob_p) # ked J je uz jedno cislo
L0g = - g * v0 * ds(tag_bc_C)

####################
## n, p Continuity
####################
znamienka = 1
if znamienka:
    a1 = -D_p * inner(grad(p), grad(v1)) * dx - mob_p * p * inner(grad(Psi), grad(v1))*dx
    L1=0

###################
if znamienka:
    a2 = -D_n * inner(grad(n), grad(v2)) * dx + mob_n * n * inner(grad(Psi), grad(v2)) * dx
    L2 = 0

###################
print("dalej")
# equilibrium:
print("equilibrium:")
bcs = [bc_psi_E, bc_n_E, bc_p_B, bc_p_C, bc_n_C]
bcs = [bc_psi_E, bc_n_E, bc_p_E, bc_p_B, bc_n_B, bc_p_C, bc_n_C]
#problem = linearproblem(a, l, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
###
F=a0+a1+a2-L0-L1-L2
problem = NonlinearProblem(F, u=u, bcs = bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
print("solve...")
aa=solver.solve(u)
print(aa)

### pokus o rozbeh :
Na2_list = np.arange(1, .2, -.2)*Na2*3
#Na = materialy(Na1, Na2_list[0], Na3, Na4)
#u.sub(1).interpolate(Na)
from myplot import myplot2Dmoj, pltsubdomains
from myplotfenics2023 import plotting, plotting_Psipn, plotting_J, plotting1D
#for Na2_ in Na2_list:
if False:
    print(Na2_, Na2)
    Na = materialy(Na1, Na2_, Na3, Na4)
    bc_p_B = dirichletbc(ScalarType(Na2_-Nd2), dofs1_bc_B, V.sub(1))
    bc_n_E = dirichletbc(ScalarType(Nd1-Na2_), dofs1_bc_E, V.sub(2))
    bcs = [bc_psi_E, bc_n_E, bc_p_E, bc_p_B, bc_n_B, bc_p_C, bc_n_C]
    F=a0+a1+a2-L0-L1-L2
    problem = NonlinearProblem(F, u=u, bcs = bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    aa=solver.solve(u)
    print(aa)
    pltret = plotting(u, Nd, Na, savename=f'aa')
###

#quickplot()
pltret = plotting(u, project(Nd,Q), project(Na,Q), savename=f'aa_equilibrium')
#myplot2Dmoj(u.sub(0), savename=f'plt2D_aa_equilibrium')
#myplot2Dmoj(u.sub(1), savename=f'plt2D_p_equilibrium')
#myplot2Dmoj(u.sub(2), savename=f'plt2D_n_equilibrium')
#plotting_J(u, savename=f"plt2D_J_equilibrium")
#pltsubdomains(Ndoping, levels=[Nd1-Na1, Nd2-Na2, Nd3-Na3, Nd4-Na4], savename=f'plt2D_doping')
#pltsubdomains(Nsubdomains, levels=[1, 2, 3, 4], savename=f'subdomains')
plotting_Psipn(u, savename=f"plt2D_Psipn_equilibrium")

Vbi = pltret["uu"][-1]
print(f"Vbi (blbost) = {Vbi:.1f} = {Psi_norma*Vbi:.2f}V")
Vbi_BE = myeval(u.sub(0), 0, YLEN) 
print(f"Vbi_BE = {Vbi_BE:.1f} = {Psi_norma*Vbi_BE:.2f}V")


####################
## RECOMBINATION ##
####################
u_eq = Function(V)
psi_eq, p_eq, n_eq = u_eq.split()
p_eq.interpolate(u.sub(1))
n_eq.interpolate(u.sub(2))

RG = (ni**2-n*p) / (Tau_p*(n+n_eq) + Tau_n*(p+p_eq))
#RGp = -(ni**2-nRG*pRG) / (Taup*(nRG+n_eq) + Taun*(pRG+p_eq)) *v1*dx # typicky model
RGp = -RG*v1*dx
RGn =  -RG*v2*dx

FRG=a0+a1+a2-L0-L1-L2-RGp-RGn
problem = NonlinearProblem(FRG, u=u, bcs = bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
aa=solver.solve(u)
print(aa)

Vbi_BE = myeval(u.sub(0), 0, YLEN) 
print(f"recomb. Vbi_BE = {Vbi_BE:.1f} = {Psi_norma*Vbi_BE:.2f}V")

#pltret = plotting(u, Nd, Na, savename=f'aa_equilibrium_RG')
pltret = plotting(u, project(Nd,Q), project(Na,Q), savename=f'aa_equilibrium_RG')
#plotting_J(u, savename=f"plt2D_J_equilibrium_RG")

####################


####################
elem_vect = VectorElement("CG", mesh.ufl_cell(), 1)
VV = FunctionSpace(mesh, elem_vect)
Jp = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth) - Vth*mob_p*ufl.grad(u.sub(1))
Jn = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth) + Vth*mob_n*ufl.grad(u.sub(2))
Jp_proj = myplot.project(Jp, VV)
Jn_proj = myplot.project(Jn, VV)

fig5 = plt.figure()
ax50 = fig5.add_subplot(211)
ax51 = fig5.add_subplot(212)
#myplot.mystreamplot(Jp_proj, ax=ax50)
#myplot.mystreamplot(Jn_proj, ax=ax51)
#fig5.savefig("bb")
####################
        
# Integral:
In_E = assemble_scalar(form(Jn_proj.sub(1) * ds(tag_bc_E)))
Ip_E = assemble_scalar(form(Jp_proj.sub(1) * ds(tag_bc_E)))
In_C = assemble_scalar(form(Jn_proj.sub(1) * ds(tag_bc_C)))
Ip_C = assemble_scalar(form(Jp_proj.sub(1) * ds(tag_bc_C)))
print(f"In_E = {In_E}")
print(f"Ip_E = {Ip_E}")
print(f"In_C = {In_C}")
print(f"Ip_C = {Ip_C}")


# plochy:
area_E = assemble_scalar(form(1 * ds(tag_bc_E)))
area_B = assemble_scalar(form(1 * ds(tag_bc_B)))
area_C = assemble_scalar(form(1 * ds(tag_bc_C)))


#######
#Psi1_eq = pltret["uu"][0]
Psi2_list = np.arange(-0, 5, 1)*1e0
Psi3_list = np.array([0, .05, .1, .2, .3, .5, .8, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 10, 12, 15, 20])*1/Psi_norma
Psi3_list = np.array([0, .05, .1, .2, .3, .5, .8, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 10])*2/Psi_norma
J_list = []
V_list = []
In_E_list = []
In_B_list = []
In_C_list = []
Ip_E_list = []
Ip_B_list = []
Ip_C_list = []
JRG_list = []
naboj_Ib = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_Ic = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_n_B = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_n_drift = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_p_B = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_p_drift = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
pltret_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
IV_Je_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
IV_Jb_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
IV_Jc_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
IV_J_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
IV_V_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
figiv, (axiv1, axiv2, axiv3) = plt.subplots(1, 3)

from data_polsko import Ic as naboj_Ic_polsko
from data_polsko import QQb as naboj_B_polsko
from data_polsko import QQc as naboj_C_polsko
from data_polsko import QQ as naboj_tot_polsko

if not os.path.isdir("results"):
    os.system("mkdir results")
#if False:
ii=0
jjj=-1
for Psi2 in Psi2_list:
#if False:
    jj=list(Psi2_list).index(Psi2)
    jjj+=1
    bc_psi_B = dirichletbc(ScalarType(Psi2), dofs0_bc_B, V.sub(0))
    for iii in range(len(Psi3_list)):
        if True:
            if jjj%2 == 0:
                ii=iii
            else:
                ii = len(Psi3_list)-iii-1
        Psi3 = Psi3_list[ii]

        print(f"bias: Psi2={Psi2}, Psi3 = {Psi3}")
        bc_psi_C = dirichletbc(ScalarType(Psi3), dofs0_bc_C, V.sub(0))
        #bcs = [bc_psi_E, bc_psi_B, bc_psi_C, bc_n_E, bc_p_B, bc_n_B, bc_p_C, bc_n_C]
        bcs = [bc_psi_E, bc_psi_B, bc_psi_C, bc_p_B, bc_n_B, bc_p_C, bc_n_C, bc_n_E]
        bcs = [bc_psi_E, bc_psi_B, bc_psi_C, bc_p_B, bc_n_B, bc_p_C, bc_n_C, bc_n_E]
        #bcs = [bc_psi_E, bc_psi_B, bc_psi_C, bc_n_B, bc_p_C, bc_n_C, bc_n_E]
        problem = NonlinearProblem(F, u=u, bcs = bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        aa=solver.solve(u)
        print(aa)
        
        #print(f"plotting {ii}, iii:{iii}, ub{jj:02d}_uc{ii:02d}")
        #pltret = plotting(u, Nd, Na, IV=[V_list,J_list], savename=f'results/aa__ub{jj:02d}_uc{iii:02d}')
        #pltret = plotting1D(u, Nd, Na, savename=f'results/bb__ub{jj:02d}_uc{iii:02d}')
        #pltret = plotting(u, project(Nd,Q), project(Na,Q), savename=f'results/bb__ub{jj:02d}_uc{iii:02d}')
        #pltret = plotting1D(u, Nd, Na, savename=f'results/cc__ub{jj:02d}_uc{ii:02d}')


        # ii: Psi3, jj: Psi2
        #if jj==4 and ii==2:
        #if jj==4 or ii==2:
        if False:
            myplot2Dmoj(u.sub(0), savename=f'results/plt2D_Psi_ub{jj:02d}_uc{ii:02d}')
            myplot2Dmoj(u.sub(1), savename=f'results/plt2D_p_ub{jj:02d}_uc{ii:02d}')
            myplot2Dmoj(u.sub(2), savename=f'results/plt2D_n_ub{jj:02d}_uc{ii:02d}')
            #plotting_J(u, subdomain_levels=[1, 2, 3, 4], subdomain_obj=Nsubdomains, savename=f"results/plt2D_J_ub{jj:02d}_uc{ii:02d}")
            plotting_J(u, subdomain_levels=[1, 2, 3, 4], subdomain_obj=None, numpoints_x=51, numpoints_y=51, savename=f"results/plt2D_J_ub{jj:02d}_uc{ii:02d}")

        #if ii == 3:
        if False:
            plotting_Psipn(u, savename=f"radek/plt2D_Psipn_ub{jj:02d}_uc{ii:02d}")

        #if ii%4 == 0:
        #if False:
        #if jj==2 or ii==4:
        #if ii == 4 or ii==1:
        if ii == 7:
            #plotting_Psipn(u, subdomains_levels=[1, 2, 3, 4], subdomains_obj=Nsubdomains, savename=f"results/plt2D_Psipn_ub{jj:02d}_uc{ii:02d}")
            #plotting_J(u, subdomains_levels=[1, 2, 3, 4], subdomains_obj=Nsubdomains, savename=f"results/plt2D_J_ub{jj:02d}_uc{ii:02d}")
            plotting_Psipn(u, savename=f"results/plt2D_Psipn_ub{jj:02d}_uc{ii:02d}")
            plotting_J(u, savename=f"results/plt2D_J_ub{jj:02d}_uc{ii:02d}")

        V_list.append(Psi3)
        Jp = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth) - Vth*mob_p*ufl.grad(u.sub(1))
        #Jn = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth) + Vth*mob_n*ufl.grad(u.sub(2))
        Jp_proj = myplot.project(Jp, VV)
        Jn_proj = myplot.project(Jn, VV)
        # Integral:
        Ip_E = assemble_scalar(form(Jp_proj.sub(1) * ds(tag_bc_E)))
        Ip_B = -assemble_scalar(form(Jp_proj.sub(1) * ds(tag_bc_B)))
        Ip_C = assemble_scalar(form(Jp_proj.sub(1) * ds(tag_bc_C)))
        #Ip_Cnn = assemble_scalar(form(inner(Jp_proj,nn) * ds(tag_bc_C)))
        Ip_E_list.append(Ip_E)
        Ip_B_list.append(Ip_B)
        Ip_C_list.append(Ip_C)
        #Ip_Cnn_list.append(Ip_Cnn)
        In_E = assemble_scalar(form(Jn_proj.sub(1) * ds(tag_bc_E)))
        In_B = -assemble_scalar(form(Jn_proj.sub(1) * ds(tag_bc_B)))
        In_C = assemble_scalar(form(Jn_proj.sub(1) * ds(tag_bc_C)))
        #In_Cnn = assemble_scalar(form(inner(Jp_proj,nn) * ds(tag_bc_C)))
        In_E_list.append(In_E)
        In_B_list.append(In_B)
        In_C_list.append(In_C)
        #print(f"In_E = {In_E}, In_B = {In_B}, In_C = {In_C}")
        #print(f'{ii}. Psi1:{Psi1_list[ii]:.2f}, {ii}. J:{J_list[ii]:.3f}')

        #pltret_list[jj][ii] = pltret

        plt.clf()

        fig, ax = plt.subplots(1, 1, figsize = (6,4))
        #ax.plot(V_list, In_C_list, ".-b")
        #ax.plot(V_list, Ip_C_list, ".-g")
        ax.plot(Psi_norma*np.array(V_list), np.array(Ip_C_list)+np.array(In_C_list), ".-k")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Collector - emitter Voltage (V)")
        ax.set_ylabel("Collector current (A)")
        ax.grid()
        plt.tight_layout()
        fig.savefig("IV_Ic")
        
        print(f"plotting {ii}, iii:{iii}, ub{jj:02d}_uc{ii:02d}")
        IV = [Psi_norma*np.array(V_list),np.array(Ip_C_list)+np.array(In_C_list)] 
        savename_bb = f'results/bb__ub{jj:02d}_uc{iii:02d}'
        savename_cc = f'results/cc__ub{jj:02d}_uc{ii:02d}'
        pltret = plotting1D(u, project(Nd, Q), project(Na, Q), savename=savename_bb, IV=IV)
        #pltret = plotting1D(u, Nd, Na, savename=savename_cc)
        os.system(f'cp {savename_bb}.png {savename_cc}.png')
        pltret_list[jj][ii] = pltret
        IV_Je_list[jj][ii] = Ip_E+In_E
        IV_Jb_list[jj][ii] = Ip_B+In_B
        IV_Jc_list[jj][ii] = Ip_C+In_C
        IV_V_list[jj][ii] = Psi3


        if False:
            # Naboje
            naboj_Ib[jj][ii] = Ip_B+In_B
            naboj_Ic[jj][ii] = Ip_C+In_C
            naboj_n_B[jj][ii] = assemble_scalar(form(u.sub(2) * dxx(tag_domain_B)))
            naboj_p_drift[jj][ii] = assemble_scalar(form(u.sub(1) * dxx(tag_domain_drift)))

            figq, (axq1, axq2) = plt.subplots(1,2,figsize=(6,4))
            axq1.plot(naboj_Ic[jj], naboj_n_B[jj], '-o', label="Base")
            axq1.plot(naboj_Ic[jj], naboj_p_drift[jj], '-og', label="Collector")
            axq2.plot(naboj_Ic_polsko[4:], naboj_B_polsko[4:], '-o', label="Base")
            axq2.plot(naboj_Ic_polsko[4:], naboj_C_polsko[4:], '-og', label="Collector")
            axq1.grid()
            axq2.grid()
            axq2.set_ylim(top=1.2)
            axq1.set_xlim(left=0)
            #axq1.legend()
            axq2.legend()
            axq1.set_title("Simulation")
            axq2.set_title("Measurement")
            axq1.set_xlabel("Collector current (A)")
            axq2.set_xlabel("Collector current (A)")
            axq1.set_ylabel("Excess charge ($\mu$C)")
            plt.tight_layout(h_pad=.2)
            figq.savefig(f'QvsIc_ub{jj}')
        #Psi3_list = Psi3_list[::-1] # odzadu
            
            fig1d, (ax1d1, ax1d2) = plt.subplots(1,2,figsize=(6,4))
            axq1.plot(naboj_Ic[jj], naboj_n_B[jj], '-o')
            axq1.plot(naboj_Ic[jj], naboj_p_drift[jj], '-o')
            axq1.grid()
            #figq.savefig(f'QvsIc_ub{jj}')


#figiv = plt.figure(figsize=(6,4))
figiv, axiv = plt.subplots(1,1,figsize=(6,6))

for jj in range(len(Psi2_list)):
    for ii in range(len(Psi3_list)):
        axiv.cla()
        savename_cc = f'results/IV/cc__ub{jj:02d}_uc{ii:02d}'
        for kk in range(len(Psi2_list)):
            axiv.plot((IV_V_list[kk]-Vbi_BE)*Psi_norma, IV_Jc_list[kk]/max(IV_Jc_list.flat), '.-k')
        axiv.plot((IV_V_list[jj, ii]-Vbi_BE)*Psi_norma, IV_Jc_list[jj,ii]/max(IV_Jc_list.flat), 'o')
        axiv.grid(True)
        axiv.set_xlabel(f"Collector - Emitter Voltage (V)")
        axiv.set_ylabel("Collector Current (Normalized)")
        axiv.set_ylim(bottom=0)
        axiv.set_ylim(top=1.2)
        axiv.set_xlim(left=0)
        axiv.set_xlim(right=20)
        figiv.savefig(savename_cc)
        

'''
PRINt("hranie sa s pltret_list...")
pltret_list

aax=pltret['axes']
ax2=aax[2]
ll = ax2.lines[0]
ll.get_xdata()
ll.get_xydata()

xdata = xdata_Vce = pltret_list[jj][ii]['axes'][3].lines[0].get_xdata()
ydata = ydata_Ic = pltret_list[jj][ii]['axes'][3].lines[0].get_ydata()
plt.clf()
plt.plot(xdata

ii=
for 


from data_polsko import Ib as naboj_Ib_polsko
from data_polsko import Qb as naboj_B_polsko
from data_polsko import Qc as naboj_C_polsko
from data_polsko import Q as naboj_tot_polsko
figq, (axq1, axq2) = plt.subplots(1,2,figsize=(6,3))
for ii in range(len(naboj_Ib[0])):
    axq1.plot(naboj_Ib[:,ii], naboj_n_B[:,ii], '-o')
    axq1.plot(naboj_Ib[:,ii], naboj_p_drift[:,ii], '-o')
    axq1.plot(naboj_Ib[:,ii], naboj_p_drift[:,ii]+naboj_n_B[:,ii], '-o')
    axq2.plot(naboj_Ib_polsko, naboj_B_polsko, '--o')
    axq2.plot(naboj_Ib_polsko, naboj_C_polsko, '--o')
    axq2.plot(naboj_Ib_polsko, naboj_tot_polsko, '--o')
    axq1.grid()
figq.savefig(f'QvsIb_uc')



if False:
    plt.clf()
    #Psi_norma=1
    plt.plot(Psi1_list*Psi_norma, J_list, ".-")
    plt.plot((Psi1_list-Psi1_eq)*Psi_norma, J_list, ".-b")
    if False:
        plt.plot(Psi1_list*Psi_norma, JRG_list, ".-r")
        plt.plot((Psi1_list-Psi1_eq)*Psi_norma, JRG_list, ".-r")
    plt.vlines([-Psi1_eq*Psi_norma], 0, max(J_list), colors='k')
    plt.grid()
    plt.savefig("IV")
    #ax=myplot.9yplot2Dmoj(u.sub(1), shading="flat")
    #ax.set_title("diery")
    #plt.savefig("aa_2D.png")
'''
