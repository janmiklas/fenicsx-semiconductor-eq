from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                         dx, grad, inner,
                         FiniteElement, VectorElement, MixedElement, TestFunctions, TrialFunctions)
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                                 form, locate_dofs_geometrical, locate_dofs_topological)
import ufl
import matplotlib.pyplot as plt
import numpy as np
import myplot
#from pn8 import mob_n, mob_p
#from params import Nd, Na, mob_n, mob_p
from params import *

################
### PLOTTING ###
################
def quickeval(uu, numpoints=10):
    ret = myplot.myevalonline(uu, [0,0], [1,0], numpoints=numpoints)[1]
    print(f"{uu}: {ret}")
    #return ret

def quickplot(u):
    fig,ax1 = plt.subplots(1, 1, sharex=True)
    (xx, yy, zz), uu = myplot.myevalonline(u, [0,0], [1,0], numpoints=1001)
    ax1.plot(xx, uu)
    fig.savefig("aa.png")

def plotting(u, Nd, Na, IV=None, savename=None):
    import matplotlib.pyplot as plt
    #fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True)
    plt.clf(); plt.cla()
    if IV:
        fig = plt.figure()
        ax00 = fig.add_subplot(421)
        ax01 = fig.add_subplot(423)
        ax02 = fig.add_subplot(425)
        ax03 = fig.add_subplot(427)
        ax1 = fig.add_subplot(122)
        axes = (ax00,ax01,ax02,ax03,ax1)
    else:
        fig, axes = plt.subplots(4,1, sharex=True)
        (ax00, ax01, ax02, ax03) = axes
    #fig.subplots_adjust(hspace=0.001)

    mesh = u.function_space.mesh

    #myplot.myplot_2d(Psi, ax, shading="flat", plotmesh=False)
    #myplot.myplot_2d(n_result, ax2, shading="flat")
    ####
    #############
    # PRUDY
    #############
    #VV = fem.VectorFunctionSpace(domain, ("CG", 1))
    #alebo:
    elem_vect = VectorElement("CG", mesh.ufl_cell(), 1)
    VV = FunctionSpace(mesh, elem_vect)
    (u0, u1, u2) = (Psi, p, n) = (Psi, p, n) =  ufl.split(u) # po tomto v pohode skonvergoval
    Vth = 0.0259 # k.T/q
    Jp = u.sub(1)*mob_p*ufl.grad(u.sub(0)) - Vth*mob_p*ufl.grad(u.sub(1))
    Jn = u.sub(2)*mob_n*ufl.grad(u.sub(0)) + Vth*mob_n*ufl.grad(u.sub(2))
    Jp_drift = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth)
    Jp_diff = -Vth*mob_p*ufl.grad(u.sub(1))
    Jn_drift = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth)
    Jn_diff = Vth*mob_n*ufl.grad(u.sub(2))
    Jp = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth) - Vth*mob_p*ufl.grad(u.sub(1))
    Jn = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth) + Vth*mob_n*ufl.grad(u.sub(2))
    E = -grad(Psi)
    #grad_p = grad(u.sub(1))
    #grad_n = grad(u.sub(2))

    #Jp_proj = myplot.project(Jp, VV)
    #Jn_proj = myplot.project(Jn, VV)
    Jp_drift_proj = myplot.project(Jp_drift, VV)
    Jp_diff_proj = myplot.project(Jp_diff, VV)
    Jn_drift_proj = myplot.project(Jn_drift, VV)
    Jn_diff_proj = myplot.project(Jn_diff, VV)
    Jn_proj = myplot.project(Jn, VV)
    Jp_proj = myplot.project(Jp, VV)
    #E_proj = myplot.project(E, VV)
    #grad_p_proj = myplot.project(grad_p, VV)
    ####
    i = 0
    linestyles = ["-", "--", ":"]
    xmin = min(mesh.geometry.x[:,0])
    xmax = max(mesh.geometry.x[:,0])
    ymin = min(mesh.geometry.x[:,1])
    ymax = max(mesh.geometry.x[:,1])
    for X in [xmin+(xmax-xmin)/2]:
        XY1 = [X, ymax]
        XY2 = [X, ymin]
        (xx, yy, zz), uu = myplot.myevalonline(u.sub(0), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uup = myplot.myevalonline(u.sub(1), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uun = myplot.myevalonline(u.sub(2), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uuNd = myplot.myevalonline(Nd, XY1, XY2, numpoints=1001)
        (xx, yy, zz), uuNa = myplot.myevalonline(Na, XY1, XY2, numpoints=1001)
        #(xx, yy, zz), uuE = myplot.myevalonline(E_proj.sub(0), XY1, XY2, numpoints=1001)
        #(xx, yy, zz), uugrad_p = myplot.myevalonline(grad_p_proj.sub(0), XY1, XY2, numpoints=1001)
        #(xx, yy, zz), uuJp = myplot.myevalonline(Jp_proj.sub(0), XY1, XY2, numpoints=1001)
        #(xx, yy, zz), uuJn = myplot.myevalonline(Jn_proj.sub(0), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uuJp = myplot.myevalonline(Jp_proj.sub(1), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uuJn = myplot.myevalonline(Jn_proj.sub(1), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uuJp_drift = myplot.myevalonline(Jp_drift_proj.sub(1), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uuJp_diff = myplot.myevalonline(Jp_diff_proj.sub(1), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uuJn_drift = myplot.myevalonline(Jn_drift_proj.sub(1), XY1, XY2, numpoints=1001)
        (xx, yy, zz), uuJn_diff = myplot.myevalonline(Jn_diff_proj.sub(1), XY1, XY2, numpoints=1001)
        uurho = uuNd-uuNa+uup-uun
        #uuJp = uuJp_drift + uuJp_diff
        #uuJn = uuJn_drift + uuJn_diff
        uuJ = uuJp + uuJn

        xx = yy
        ax00.plot(xx, uu, linestyle=linestyles[i])
        #ax00.plot(xx, uuE)
        #ax00.plot(xx, uugrad_p)
        ax01.plot(xx, uun, "b", linestyle=linestyles[i])
        ax01.plot(xx, uup, "g", linestyle=linestyles[i])
        ax01.plot(xx, uuNd, "b--")
        ax01.plot(xx, uuNa, "g--")
        ax01.plot(xx, uurho, "k")
        #ax01.set_ylim([1e-3,1e1])
        ax02.plot(xx, uuJp_drift, "g--")
        ax02.plot(xx, uuJp_diff, "g:")
        ax02.plot(xx, uuJn_drift, "b--")
        ax02.plot(xx, uuJn_diff, "b:")
        ax03.plot(xx, uuJp, "g")
        ax03.plot(xx, uuJn, "b")
        ax03.plot(xx, uuJ, "k")
        #ax03.plot(xx, uuJp_diff, "g:")
        #ax03.plot(xx, uuJn_diff, "b:")
        if IV:
            V = list(IV[0])+[uu[-1]]
            I = list(IV[1])+[uuJ[-1]]
            ax1.plot(V, I, ".-")
            ax1.plot(V[-1], I[-1], "o")
            ax1.grid()
            ax1.set_xlabel("Uce")
            ax1.set_ylabel("Ic")
        i+=1

    ax00.set_ylabel("Psi")
    ax01.set_ylabel("p, n")
    ax02.set_ylabel("Jp, Jn")
    ax03.set_ylabel("J, Jp, Jn")
    ax00.grid()
    ax01.grid()
    ax02.grid()
    ax03.grid()
    #ax01.semilogy()
    #ax00.set_ylim([-10, 60])
    #ax00.set_ylim([-10, 10])
    #ax00.set_ylim([-20e-2, 10e-2])
    #ax01.set_ylim([-1, 2])
    #fig.suptitle(f'ldkjds') # title celeho figure

    #ax01.set_yticklabels(ax01.get_yticklabels()[1:-1])
    #ax02.set_yticklabels(ax02.get_yticklabels()[1:-1])
    #ax03.set_yticklabels(ax03.get_yticklabels()[1:-1])
    for axx in axes:
        axx.set_yticks(axx.get_yticks()[1:-1])

    #plt.savefig("aa.png")
    if savename:
        plt.savefig(savename)
        plt.savefig("aa.png")

    #myplot.myplot_2d(Nd, ax, shading="flat", savename="aa.png")
    #return ((xx, yy, zz), uu, uup, uun, uuNd, uuNa, uurho)
    return {"x":(xx,yy,zz), "uu":uu, "uuJ":uuJ, "uup":uup, "uun":uun, "Jp_proj":Jp_proj, "Jn_proj":Jn_proj, "axes":axes, "fig":fig}


def plotting_J(u, fig=None, savename=None, numpoints_x=11, numpoints_y=11, aspect_equal=True):
    
    import matplotlib.pyplot as plt
    #fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True)
    #plt.clf(); plt.cla()
    if not fig:
        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    fig.subplots_adjust(hspace=0)

    mesh = u.function_space.mesh

    #myplot.myplot_2d(Psi, ax, shading="flat", plotmesh=False)
    #myplot.myplot_2d(n_result, ax2, shading="flat")
    ####
    #############
    # PRUDY
    #############
    #VV = fem.VectorFunctionSpace(domain, ("CG", 1))
    #alebo:
    elem_vect = VectorElement("CG", mesh.ufl_cell(), 1)
    VV = FunctionSpace(mesh, elem_vect)
    (u0, u1, u2) = (Psi, p, n) = (Psi, p, n) =  ufl.split(u) # po tomto v pohode skonvergoval
    Vth = 0.0259 # k.T/q
    #Jp = u.sub(1)*mob_p*ufl.grad(u.sub(0)) - Vth*mob_p*ufl.grad(u.sub(1))
    #Jn = u.sub(2)*mob_n*ufl.grad(u.sub(0)) + Vth*mob_n*ufl.grad(u.sub(2))
    #Jp_drift = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth)
    #Jp_diff = -Vth*mob_p*ufl.grad(u.sub(1))
    #Jn_drift = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth)
    #Jn_diff = Vth*mob_n*ufl.grad(u.sub(2))
    Jp = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth) - Vth*mob_p*ufl.grad(u.sub(1))
    Jn = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth) + Vth*mob_n*ufl.grad(u.sub(2))

    #Jp_drift_proj = myplot.project(Jp_drift, VV)
    #Jp_diff_proj = myplot.project(Jp_diff, VV)
    #Jn_drift_proj = myplot.project(Jn_drift, VV)
    #Jn_diff_proj = myplot.project(Jn_diff, VV)
    Jn_proj = myplot.project(Jn, VV)
    Jp_proj = myplot.project(Jp, VV)

    #################
    # streamlines
    #################
    xmax = max(mesh.geometry.x[:,0])
    ymax = max(mesh.geometry.x[:,1])
    print("xmax ymax hotovo")
    xx=np.linspace(0, xmax, numpoints_x)
    yy=np.linspace(0, ymax, numpoints_y)
    print("xx yy hotovo")
    #xv, yv = np.meshgrid(xx, yy)
    Jpx=np.array([[myplot.myeval(Jp_proj.sub(0),X,Y) for X in xx] for Y in yy])
    print("myeval x hotovo")
    Jpy=np.array([[myplot.myeval(Jp_proj.sub(1),X,Y) for X in xx] for Y in yy])
    print("myeval y hotovo")
    Jnx=np.array([[myplot.myeval(Jn_proj.sub(0),X,Y) for X in xx] for Y in yy])
    Jny=np.array([[myplot.myeval(Jn_proj.sub(1),X,Y) for X in xx] for Y in yy])
    Jx = Jpx+Jnx
    Jy = Jpy+Jny
    
    ax1.streamplot(xx, yy, Jpx, Jpy, density=.5, color="g", broken_streamlines=False)
    print("streamplot Jp hotovo")
    ax2.streamplot(xx, yy, Jnx, Jny, density=.5, color="b", broken_streamlines=False)
    print("streamplot Jn hotovo")
    ax3.streamplot(xx, yy, Jpx, Jpy, density=.5, color="g", broken_streamlines=False)
    ax3.streamplot(xx, yy, Jnx, Jny, density=.5, color="b", broken_streamlines=False)
    ax3.streamplot(xx, yy, Jx, Jy, density=.5, color="k", broken_streamlines=False)
    print("streamplot J hotovo")

    if savename:
        fig.savefig(savename)
    
    return (ax1, ax2, ax3), xx,yy, (Jpx, Jnx, Jx), (Jpy, Jny, Jy)
