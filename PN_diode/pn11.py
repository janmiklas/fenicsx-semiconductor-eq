import gmsh
import numpy as np
#import pyvista
import matplotlib.pyplot as plt
from myplot import myplot_2d, pltsubdomains, myplot2Dmoj, project
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
    n=(4, 1000),
    cell_type=CellType.triangle,
)

tag_bc_E = 28
tag_bc_B = 29
tag_bc_C = 30
tag_bc_medzera = 31

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
(u0, u1, u2) = (Psi, p, n) = (Psi, p, n) =  ufl.split(u) # po tomto v pohode skonvergoval
#u = TrialFunction(V)
#u0 = Psi = u[0]
#u1i = p = u[1]
#u2 = n = u[2]
#(u0, u1, u2) = (Psi, p, n) = TrialFunctions(V)

###########################
### Discontinuous kappa ###
###########################

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
def doping_expexp(dist, C0, Clen, exponent=10):
    #return C0 * np.exp(-(np.exp(dist/6)-1)**6)
    return C0 * np.exp(-(np.exp(dist/1.44/Clen)-1)**exponent)
#######################################3


def dop4(x, Yab=0, Xa=0, Xb=XLEN, C0=Na4, Clen=YLEN):
    d = distance(x, Yab=Yab, Xa=Xa, Xb=Xb)
    return doping_constant(d, C0, Clen)

def dop2(x, Yab=YLEN, Xa=0, Xb=XLEN, C0=Nd2, Clen=.5*YLEN):
    d = distance(x, Yab=Yab, Xa=Xa, Xb=Xb)
    #return doping_constant(d, C0, Clen)
    #return doping_log10(d, C0, Clen)
    #return doping_gauss(d, C0, Clen, exponent=10)
    return doping_expexp(d, C0, Clen, exponent=10)

#doping_P = Na4
doping_P = Function(Q)
doping_N = Function(Q)
#doping_P.interpolate(dop4)
# alebo:
#doping_P.interpolate(lambda x: np.full(x.shape[1], Na4, dtype=np.float64))
# alebo:
#doping_P.vector.array[:] = Na4
# alebo:
doping_P.x.array[:] = Na4
doping_N.interpolate(dop2)

Nd = doping_N
Na = doping_P
N = Nd-Na
N_proj = project(N, Q)

if not os.path.isdir("results"):
    os.system("mkdir results")
myplot2Dmoj(N_proj, savename="doping", aspect_equal=False)

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

#print("plot pinit...")
#myplot2Dmoj(pinit, savename="results/pinit")
#print("plot ninit...")
#myplot2Dmoj(ninit, savename="results/ninit")

#n_i1.interpolate(Nd)
#p_i1.interpolate(Na)
#space1, map1 = V.sub(1).collapse(collapsed_dofs=True)
#u.sub(1).interpolate(Na)

u.sub(1).interpolate(pinit)
u.sub(2).interpolate(ninit)
#u.sub(0).interpolate(Psiinit)

#####################
## poriadny sposob 

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
    return np.isclose(x[1], YLEN)
facets_bc_E = facets = locate_entities_boundary(mesh, mesh.topology.dim-1, contact_E) # alebo rovnako aj locate_entities (to len tie entity vrati v inom poradi). locate_entities ich mimochodom vracia zoradene 
#tag=tag_bc_E
#facets = facets_bc_E = ft.indices[ft.values==tag]
dofs0 = dofs0_bc_E = locate_dofs_topological(V, mesh.topology.dim-1, facets)
dofs1 = locate_dofs_topological(V.sub(1), mesh.topology.dim-1, facets)
dofs2 = locate_dofs_topological(V.sub(2), mesh.topology.dim-1, facets)
bc_psi_E = dirichletbc(ScalarType(Psi1), dofs0, V.sub(0))
bc_p_E = dirichletbc(ScalarType(0), dofs1, V.sub(1))
bc_n_E = dirichletbc(ScalarType(Nd2-Na4), dofs2, V.sub(2))

def contact_C(x):
    return np.isclose(x[1], 0)
facets_bc_C = facets = locate_entities_boundary(mesh, mesh.topology.dim-1, contact_C)
#tag=tag_bc_C
#facets = facets_bc_C = ft.indices[ft.values==tag]
dofs0 = dofs0_bc_C = locate_dofs_topological(V, mesh.topology.dim-1, facets)
dofs1 = locate_dofs_topological(V.sub(1), mesh.topology.dim-1, facets)
dofs2 = locate_dofs_topological(V.sub(2), mesh.topology.dim-1, facets)
bc_psi_C = dirichletbc(ScalarType(Psi3), dofs0, V.sub(0))
bc_p_C = dirichletbc(ScalarType(Na4), dofs1, V.sub(1))
bc_n_C = dirichletbc(ScalarType(0), dofs2, V.sub(2))

# https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
facet_markers_list = [tag_bc_E, tag_bc_C]
facet_indices = [facets_bc_E, facets_bc_C]

facet_markers = [np.full_like(facet_indices[i], facet_markers_list[i]) for i in range(len(facet_markers_list))]
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tags = meshtags(mesh, mesh.topology.dim-1, facet_indices[sorted_facets], facet_markers[sorted_facets])


ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tags)  
dS = ufl.Measure('dS', domain=mesh, subdomain_data=facet_tags)  


###############
### POISSON ###
###############

#a = inner(-kappa*grad(u), grad(v)) * dx
a0 = inner(grad(Psi), grad(v0)) * dx ### POZOR tuto je to minus nejake pofiderne, ale s nim to funguje. mozno zalezi od znamienka elementarneho naboja
#L = Constant(mesh, ScalarType(10)) * v * dx
L0 = 1/lmda0 * (Nd-Na-n+p + EPS) * v0 * dx # EPS aby tam nevznikla nula
#L0 = -1* (Nd-Na-n+p + EPS) * v0 * dx # EPS aby tam nevznikla nula

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
    #a1 = p*v1*dx + dt*D_p*inner(grad(p), grad(v1))*dx
    #L1 = p_*v1*dx - dt*mob_p*p_*inner(grad(Psi_), grad(v1))*dx
    #a1 = -D_p * inner(grad(p), grad(v1)) * dx
    #L1 = mob_p * p * inner(grad(Psi), grad(v1))*dx
    a1 = -D_p * inner(grad(p), grad(v1)) * dx - mob_p * p * inner(grad(Psi), grad(v1))*dx
    L1=0

###################
if znamienka:
    #a2 = n*v2*dx + dt*D_n*inner(grad(n), grad(v2))*dx
    #L2 = n_*v2*dx + dt*mob_n*n_*inner(grad(Psi_), grad(v2))*dx
    #a2 = -D_n * inner(grad(n), grad(v2)) * dx
    #L2 = -mob_n * n * inner(grad(Psi), grad(v2)) * dx
    a2 = -D_n * inner(grad(n), grad(v2)) * dx + mob_n * n * inner(grad(Psi), grad(v2)) * dx
    L2 = 0

###################
print("dalej")
a = a0 + a1 + a2
L = L0 + L1 + L2 
# equilibrium:
print("equilibrium:")
bcs = [bc_psi_C, bc_n_C]
bcs = [bc_psi_C, bc_n_C, bc_p_E]
bcs = [bc_psi_E, bc_n_C, bc_p_C, bc_n_E, bc_p_E]
#bcs = [bc_psi_E, bc_p_C, bc_n_E]
#bcs = [bc_psi_C, bc_p_C, bc_n_E]
#problem = linearproblem(a, l, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
###
F=a0+a1+a2-L0-L1-L2
problem = NonlinearProblem(F, u=u, bcs = bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
print("solve...")
aa=solver.solve(u)
print(aa)

#quickplot()
#from myplotlocal import plotting, plotting_J
from myploteeict import plotting, plotting_J
from myplot import myplot2Dmoj
pltret = plotting(u, Nd, Na, savename=f'results/aa_equilibrium')
pltret = plotting(u, Nd, Na, savename=f'aa')
#myplot2Dmoj(u.sub(0), savename=f'plt2D_aa_equilibrium')
#myplot2Dmoj(u.sub(1), savename=f'plt2D_p_equilibrium')
#myplot2Dmoj(u.sub(2), savename=f'plt2D_n_equilibrium')


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

#FRG=a00+a11+a22-L00-L11-L22-RGp-RGn
####################

print("Prudova BC")
J = 1e-10
#J_list = np.arange(0, 1e-0, 1e-2)
J_list =  np.array([0, 1, 5, 10, 50, 100, 500, 1000])*3e-3
J_list =  np.array([0, 0.2, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80])*1e-5
Jorig_list =  np.array([0, 0.2, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80])*2e6
Jorig_list = (np.exp(np.arange(0,10))-1)**1*2e-1
J_list = Jorig_list/lmdaJ
PsiC_list = []
PsiE_list = []
V_list = []
uuJ_list = []
#Jc_list = []
#Je_list = []
Ic_list = []
Ie_list = []

elem_vect = VectorElement("CG", mesh.ufl_cell(), 1)
VV = FunctionSpace(mesh, elem_vect)
#J_list=[]
for J in J_list:
    ii = list(J_list).index(J)
    bcs = [bc_psi_E, bc_p_C, bc_n_E, bc_p_E, bc_n_C]
    #bcs = [bc_psi_E, bc_p_C, bc_n_E, bc_p_E]
    #bcs = [bc_psi_E, bc_n_E, bc_p_E]
    #g = -J/p/mob_p
    #g = (-Dp*inner(grad(p), nn) - J)/(p*mob_p) # ked J je uz jedno cislo
    g = (Dn*inner(grad(n), nn) -Dp*inner(grad(p), nn) - J)/(mob_n*n + mob_p*p)
    #g = (inner((Dn*grad(n) -Dp*grad(p)), nn) - J)/(mob_n*n + mob_p*p)
    #gN=0
    L0g = - g * v0 * ds(tag_bc_C)
    F=a0+a1+a2-L0-L1-L2-L0g - RGp-RGn
    problem = NonlinearProblem(F, u=u, bcs = bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    print(Jorig_list[ii], J)
    aa=solver.solve(u)
    print(aa)
    #Integral:
    Jp = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth) - Vth*mob_p*ufl.grad(u.sub(1))
    Jn = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth) + Vth*mob_n*ufl.grad(u.sub(2))
    Jp_proj = myplot.project(Jp, VV)
    Jn_proj = myplot.project(Jn, VV)
    area_E = assemble_scalar(form(1 * ds(tag_bc_E)))
    area_C = assemble_scalar(form(1 * ds(tag_bc_C)))
    In_E = assemble_scalar(form(Jn_proj.sub(1) * ds(tag_bc_E)))
    Ip_E = assemble_scalar(form(Jp_proj.sub(1) * ds(tag_bc_E)))
    In_C = assemble_scalar(form(Jn_proj.sub(1) * ds(tag_bc_C)))
    Ip_C = assemble_scalar(form(Jp_proj.sub(1) * ds(tag_bc_C)))
    Ic_list.append(Ip_C+In_C)
    Ie_list.append(Ip_E+In_E)
    Jc_list = np.array(Ic_list)/area_C
    Je_list = np.array(Ie_list)/area_E

    V_list.append(myplot.myeval(u.sub(0), XLEN/2, 0))
    if J == 0:
        #Vbi = pltret["uu"][-1]
        Vbi = V_list[-1]
        print(f"Vbi = {Vbi:.1f} = {Psi_norma*Vbi:.2f}V")
    
    #pltret = plotting(u, Nd, Na, savename=f'results/bb{ii:02d}')
    IV = [(np.array(V_list)-Vbi)*Psi_norma, Je_list*lmda_J]
    #pltret = plotting(u, Nd, Na, IV=IV, savename=f'results/bb{ii:02d}')
    pltret = plotting(u, Nd, Na, IV=None, savename=f'results/bb{ii:02d}')
    PsiC_list.append(pltret["uu"][-1])
    #PsiE_list.append(pltret["uu"][0])
    uuJ_list.append(pltret["uuJ"][0])
plt.clf()
fig = plt.figure(figsize=(4,4))
#scale_J=0.007/2
scale_J=1/lmdaJ
#plt.plot(Psi_norma*(np.array(PsiC+list), 1*np.array(J_list), '-o', label="J")
#plt.plot(Psi_norma*(np.array(PsiC_list)-Vbi), 1*np.array(Jc_list), '-o', label="Jc")
#plt.plot(Psi_norma*(np.array(PsiC_list)-Vbi), 1*np.array(Je_list), '-o', label="Je")
#plt.plot(Psi_norma*(np.array(PsiC_list)-Vbi), 1*np.array(uuJ_list), '-o', label="uuJ")
plt.plot(Psi_norma*(np.array(PsiC_list)-Vbi), 1*np.array(Je_list)*scale_J/max(Je_list), '-o', label="Je")
plt.plot(Psi_norma*(np.array(PsiC_list)-Vbi), 1*np.array(uuJ_list)*scale_J/max(uuJ_list), '--o', label="Je")
plt.plot(Psi_norma*(np.array(PsiC_list)-Vbi), 1*np.array(J_list)*scale_J/max(J_list), ':o', label="Je")
plt.xlabel("Voltage (V)")
plt.ylabel("Forward current (A)")
#plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.savefig("IV")
plt.savefig("results/IV")
plt.savefig("results/IV.pdf")

print(np.round(np.array(Jc_list)/Jc_list[1]))



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



#figiv = plt.figure(figsize=(6,4))
figiv, axiv = plt.subplots(1,1,figsize=(4,4))
IV_V_list = PsiC_list
IV_Ic_list = np.array(Ic_list)/max(Ic_list) * max(uuJ_list)

for ii in range(len(IV_Ic_list)):
    axiv.cla()
    savename_cc = f'results/IV/cc_{ii:02d}'
    axiv.plot((IV_V_list-Vbi)*Psi_norma, IV_Ic_list, '.-k')
    axiv.plot((IV_V_list[ii]-Vbi)*Psi_norma, IV_Ic_list[ii], 'o')
    axiv.grid(True)
    axiv.set_xlabel(f"Voltage (Anode - Cathode) (V)")
    axiv.set_ylabel("Current (Normalized)")
    axiv.set_ylim(bottom=0)
    #axiv.set_ylim(top=1.2)
    axiv.set_xlim(left=0)
    #axiv.set_xlim(right=20)
    figiv.savefig(savename_cc)
