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
from myplot import mesh2triang, myevalonpoints

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


def plotting(u, Nd, Na, IV=None, savename=None, figsize=(6,4)):
    import matplotlib.pyplot as plt
    #fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True)
    plt.clf(); plt.cla()
    if IV:
        fig = plt.figure(figsize=(2*figsize[0], figsize[1]))
        ax00 = fig.add_subplot(421)
        ax01 = fig.add_subplot(423)
        ax02 = fig.add_subplot(425)
        ax03 = fig.add_subplot(427)
        ax1 = fig.add_subplot(122)
        axes = (ax00,ax01,ax02,ax03,ax1)
    else:
        fig, axes = plt.subplots(4,1, sharex=True, figsize=figsize)
        (ax00, ax01, ax02, ax03) = axes
    #fig.subplots_adjust(hspace=0)

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
        xx=xx*X_norma/100/1e-6 # naspat do mikronov
        xx=-xx+xx[0]
        scale_J = 0.007
        for uuJJ in [uuJp_drift, uuJp_diff, uuJp, uuJn_drift, uuJn_diff, uuJn, uuJ]:
            uuJJ /= scale_J

        ax00.plot(xx, uu*Psi_norma, linestyle=linestyles[i])
        #ax00.plot(xx, uuE)
        #ax00.plot(xx, uugrad_p)
        ax01.plot(xx, uun, "b", label="$n$")
        ax01.plot(xx, uup, "g", label="$p$" )
        ax01.plot(xx, uuNd, "b--", label="$N_D$")
        ax01.plot(xx, uuNa, "g--", label="$N_A$")
        ax01.plot(xx, uurho, "k", label="Space charge")
        #ax01.set_ylim([1e-3,1e1])
        ax02.plot(xx, uuJp_drift, "g", label="$p$ drift")
        ax02.plot(xx, uuJp_diff, "g--", label="$p$ diffusion")
        ax02.plot(xx, uuJn_drift, "b", label="$n$ drift")
        ax02.plot(xx, uuJn_diff, "b--", label="$n$ diffusion")
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

    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    #ax03.set_xlabel(r'$x \,\, \mathrm{(\mu m)}$')
    #ax00.set_ylabel(r"$\psi$ (V)")
    ax03.set_xlabel('Position coordinate $y \mathrm{(\mu m)}$')
    ax00.set_ylabel("Potential \n (V)")
    ax01.set_ylabel(f"Concentration \n ({N_norma}/"+"$\mathrm{cm^3)}$")
    ax02.set_ylabel("Current \n (A)")
    ax03.set_ylabel("Current \n (A)")
    ax00.grid()
    ax01.grid()
    ax02.grid()
    ax03.grid()
    #ax01.semilogy()
    #ax00.set_ylim([-1, 1])
    #ax00.set_ylim([-10, 10])
    #ax00.set_ylim([-20e-2, 10e-2])
    #ax01.set_ylim([-1, 2])
    #ax03.set_ylim([0, 1])
    #fig.suptitle(f'ldkjds') # title celeho figure


    plt.savefig("aa.png")
    if savename:
        plt.tight_layout(h_pad=.2)
        plt.savefig(savename)
        #plt.savefig(savename+".pdf")
        #plt.savefig("aa.png")

    #myplot.myplot_2d(Nd, ax, shading="flat", savename="aa.png")
    #return ((xx, yy, zz), uu, uup, uun, uuNd, uuNa, uurho)
    return {"x":(xx,yy,zz), "uu":uu, "uuJ":uuJ, "uup":uup, "uun":uun, "Jp_proj":Jp_proj, "Jn_proj":Jn_proj, "axes":axes, "fig":fig}


def plotting1D(u, Nd, Na, IV=None, figsize = (6,4), savename=None):
    import matplotlib.pyplot as plt
    #fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True)
    plt.clf(); plt.cla()
    if IV:
        fig = plt.figure(figsize=(2*figsize[0], figsize[1]))
        ax00 = fig.add_subplot(221)
        ax01 = fig.add_subplot(223)
        ax1 = fig.add_subplot(122)
        axes = (ax00,ax01,ax1)
    else:
        fig, axes = plt.subplots(2,1, sharex=True, figsize=figsize)
        (ax00, ax01) = axes
    #fig.subplots_adjust(hspace=0)

    mesh = u.function_space.mesh

    #myplot.myplot_2d(Psi, ax, shading="flat", plotmesh=False)
    #myplot.myplot_2d(n_result, ax2, shading="flat")
    #############
    #VV = fem.VectorFunctionSpace(domain, ("CG", 1))
    #alebo:
    elem_vect = VectorElement("CG", mesh.ufl_cell(), 1)
    VV = FunctionSpace(mesh, elem_vect)
    (u0, u1, u2) = (Psi, p, n) = (Psi, p, n) =  ufl.split(u) # po tomto v pohode skonvergoval
    Vth = 0.0259 # k.T/q

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
        uurho = uuNd-uuNa+uup-uun

        xx = yy
        #xx=xx*X_norma/100/1e-6 # naspat do mikronov
        xx=-xx+xx[0]

        ax00.plot(xx, uu*Psi_norma, linestyle=linestyles[i])
        ax01.plot(xx, uun, "b", label="$n$")
        ax01.plot(xx, uup, "g", label="$p$" )
        ax01.plot(xx, uuNd, "b--", label="$N_D$")
        ax01.plot(xx, uuNa, "g--", label="$N_A$")
        ax01.plot(xx, uurho, "k", label="Space charge")

    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    ax01.set_xlabel('Position coordinate $y \mathrm{(\mu m)}$')
    ax00.set_ylabel("Potential \n (V)")
    ax01.set_ylabel(f"Concentration \n ({N_norma}/"+"$\mathrm{cm^3)}$")
    ax00.grid()
    ax01.grid()
    #ax01.semilogy()
    #ax01.set_ylim(top=1)
    #ax00.set_ylim(top=4)
    #ax00.set_xlim([0, 30])
    ax00.set_ylim([-.1, 4])
    #ax01.set_ylim([-.5, 1])
    ax01.set_ylim([-.2, .2])
    #fig.suptitle(f'ldkjds') # title celeho figure


    #plt.savefig("aa.png")
    if savename:
        plt.tight_layout(h_pad=.2)
        plt.savefig(savename)
        #plt.savefig(savename+".pdf")
        #plt.savefig("aa.png")

    #myplot.myplot_2d(Nd, ax, shading="flat", savename="aa.png")
    #return ((xx, yy, zz), uu, uup, uun, uuNd, uuNa, uurho)
    return {"x":(xx,yy,zz), "uu":uu, "uup":uup, "uun":uun, "axes":axes, "fig":fig}


def plotting_J(u, fig=None, savename=None, numpoints_x=11, numpoints_y=11, subdomains_obj=None, subdomains_levels=None, aspect_equal=False, figsize = (6, 6)):
    
    import matplotlib.pyplot as plt
    #fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True, sharey=True)
    #plt.clf(); plt.cla()
    
    if not fig:
        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, sharey=True, figsize=figsize)
    #fig.subplots_adjust(hspace=0)

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
    xmin = min(mesh.geometry.x[:,0])
    xmax = max(mesh.geometry.x[:,0])
    ymin = min(mesh.geometry.x[:,1])
    ymax = max(mesh.geometry.x[:,1])
    xx=np.linspace(xmin, xmax, numpoints_x)
    yy=np.linspace(ymin, ymax, numpoints_y)
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
    #ax3.streamplot(xx, yy, Jpx, Jpy, density=.5, color="g", broken_streamlines=False)
    #ax3.streamplot(xx, yy, Jnx, Jny, density=.5, color="b", broken_streamlines=False)
    ax3.streamplot(xx, yy, Jx, Jy, density=.5, color="k", broken_streamlines=False)
    print("streamplot J hotovo")
    
    if subdomains_levels:
        mt = mesh2triang(mesh)
        #C = obj.vector.array
        points = [[mt.x[i],mt.y[i]] for i in range(len(mt.x))]
        C =myevalonpoints(subdomains_obj, points)
        levels = np.sort(subdomains_levels)*.999
        CS = ax1.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='red')
        CS = ax2.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='red')
        CS = ax3.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='red')

    ax1.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax2.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax3.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax3.set_xlabel('X coordinate ($\mathrm{\mu m}$)')
    ax1.set_title("Hole Current")
    ax2.set_title("Electron Current")
    ax3.set_title("Total Current")

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    if savename:
        fig.savefig(savename)
    
    return (ax1, ax2, ax3), xx,yy, (Jpx, Jnx, Jx), (Jpy, Jny, Jy)


def plotting_Psipn(u, fig=None, savename=None, subdomains_obj=None, subdomains_levels=None, figsize = (6, 6), aspect_equal=False):
    
    import matplotlib.pyplot as plt
    #plt.clf(); plt.cla()
    
    if not fig:
        fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True, sharey=True, figsize=figsize)
    #fig.subplots_adjust(hspace=0)

    mesh = u.function_space.mesh
    mt = mesh2triang(mesh)
    #C = obj.vector.array
    points = [[mt.x[i],mt.y[i]] for i in range(len(mt.x))]
    C0 =myevalonpoints(u.sub(0), points)
    C1 =myevalonpoints(u.sub(1), points)
    C2 =myevalonpoints(u.sub(2), points)
    plotcolor=True; plotcontour=True; num_contours=10; clabels=True; shading='gouraud'
    if plotcolor:
        ax0.tripcolor(mt, C0, shading=shading) # shading= 'flat' alebo 'gouraud'
        ax1.tripcolor(mt, C1, shading=shading) # shading= 'flat' alebo 'gouraud'
        ax2.tripcolor(mt, C2, shading=shading) # shading= 'flat' alebo 'gouraud'
    if plotcontour:
            CS0 = ax0.tricontour(mt, C0, num_contours,linewidths=1, colors='k')
            CS1 = ax1.tricontour(mt, C1, num_contours,linewidths=1, colors='k')
            CS2 = ax2.tricontour(mt, C2, num_contours,linewidths=1, colors='k')
            if clabels:
                ax0.clabel(CS0, inline=1)
                ax1.clabel(CS1, inline=1)
                ax2.clabel(CS2, inline=1)
    
    if subdomains_levels:
        mt = mesh2triang(mesh)
        #C = obj.vector.array
        points = [[mt.x[i],mt.y[i]] for i in range(len(mt.x))]
        C =myevalonpoints(subdomains_obj, points)
        levels = np.sort(subdomains_levels)*.999
        CS = ax0.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='orange')
        CS = ax1.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='orange')
        CS = ax2.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='orange')

    ax0.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax1.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax2.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax2.set_xlabel('X coordinate ($\mathrm{\mu m}$)')
    ax0.set_title("Electric Potential")
    ax1.set_title("Concentration of Holes")
    ax2.set_title("Concentration of Electrons")

    xmin = min(mesh.geometry.x[:,0])
    xmax = max(mesh.geometry.x[:,0])
    ymin = min(mesh.geometry.x[:,1])
    ymax = max(mesh.geometry.x[:,1])
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)

    if savename:
        fig.savefig(savename)
    
    return (ax0, ax1, ax2)

def plotting_PsipnIV(u, fig=None, savename=None, subdomains_obj=None, subdomains_levels=None, figsize = (6, 6), aspect_equal=False):
    
    import matplotlib.pyplot as plt
    #plt.clf(); plt.cla()
    
    if not fig:
        fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True, sharey=True, figsize=figsize)
    #fig.subplots_adjust(hspace=0)

    mesh = u.function_space.mesh
    mt = mesh2triang(mesh)
    #C = obj.vector.array
    points = [[mt.x[i],mt.y[i]] for i in range(len(mt.x))]
    C0 =myevalonpoints(u.sub(0), points)
    C1 =myevalonpoints(u.sub(1), points)
    C2 =myevalonpoints(u.sub(2), points)
    plotcolor=True; plotcontour=True; num_contours=10; clabels=True; shading='gouraud'
    if plotcolor:
        ax0.tripcolor(mt, C0, shading=shading) # shading= 'flat' alebo 'gouraud'
        ax1.tripcolor(mt, C1, shading=shading) # shading= 'flat' alebo 'gouraud'
        ax2.tripcolor(mt, C2, shading=shading) # shading= 'flat' alebo 'gouraud'
    if plotcontour:
            CS0 = ax0.tricontour(mt, C0, num_contours,linewidths=1, colors='k')
            CS1 = ax1.tricontour(mt, C1, num_contours,linewidths=1, colors='k')
            CS2 = ax2.tricontour(mt, C2, num_contours,linewidths=1, colors='k')
            if clabels:
                ax0.clabel(CS0, inline=1)
                ax1.clabel(CS1, inline=1)
                ax2.clabel(CS2, inline=1)
    
    if subdomains_levels:
        mt = mesh2triang(mesh)
        #C = obj.vector.array
        points = [[mt.x[i],mt.y[i]] for i in range(len(mt.x))]
        C =myevalonpoints(subdomains_obj, points)
        levels = np.sort(subdomains_levels)*.999
        CS = ax0.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='orange')
        CS = ax1.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='orange')
        CS = ax2.tricontour(mt, C,levels=levels,linewidths=1, linestyles='dashed', colors='orange')

    ax0.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax1.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax2.set_ylabel('Y coordinate ($\mathrm{\mu m}$)')
    ax2.set_xlabel('X coordinate ($\mathrm{\mu m}$)')
    ax0.set_title("Electric Potential")
    ax1.set_title("Concentration of Holes")
    ax2.set_title("Concentration of Electrons")

    xmin = min(mesh.geometry.x[:,0])
    xmax = max(mesh.geometry.x[:,0])
    ymin = min(mesh.geometry.x[:,1])
    ymax = max(mesh.geometry.x[:,1])
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)

    if savename:
        fig.savefig(savename)
    
    return (ax0, ax1, ax2)
