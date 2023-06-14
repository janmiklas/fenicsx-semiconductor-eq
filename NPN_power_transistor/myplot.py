#from dolfinx import *
import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from time import sleep
import matplotlib.tri as tri
# pre funkciu project:
from ufl import TrialFunction, TestFunction, inner, dx
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import Expression, Function

from dolfinx.cpp.mesh import Mesh

plt.style.use('classic')


def my_eval_on_line(u, V, xx, yy):
    ## kontrola 'isiterable':
    if not hasattr(xx, '__iter__'):
        xx = [xx]
    if not hasattr(yy, '__iter__'):
        yy = [yy]
    #uu = interpolate(u, V)
    uu = project(u, V)
    return(np.array([uu(X,Y) for X in xx for Y in yy]))

def myeval__stare(u, X, Y):
    import dolfinx.geometry
    mesh = u.function_space.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, 2)
    point = [X, Y, 0]
    cell_candidates = dolfinx.geometry.compute_collisions_point(bb_tree, point)
    cell = dolfinx.geometry.select_colliding_cells(mesh, cell_candidates, point, 1)
    return u.eval(point, cell)[0]

def myeval(u, X, Y):
    tol = 0.001
    u_values = []
    import dolfinx.geometry
    mesh = u.function_space.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, 2)
    cells = []
    point = [X, Y, 0]
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, point)
    cell = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point)
    cell = cell[0] # jedna z nich
    return u.eval(point, cell)[0]

def myevalonpoints(u, points_xy):
    tol = 0.001
    u_values = []
    import dolfinx.geometry
    mesh = u.function_space.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, 2)
    cells = []
    points = []
    for point in points_xy:
        point = [point[0], point[1], 0]
        #cell_candidates = dolfinx.geometry.compute_collisions_point(bb_tree, point)
        #cell = dolfinx.geometry.select_colliding_cells(mesh, cell_candidates, point, 1)
        cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, point)
        cell = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point)
        cells.append(cell[0])
        points.append(point)
    #u_values = u.eval(points_xy, cells)
    u_values = u.eval(points, cells)
    return u_values.T[0]

def myevalonpoints_xxyy(u, points_xxyy):
    #points_xxyy vo forme [[x1,x2,x3,x4,...],[y1,y2,y3,y4,...]]
    tol = 0.001
    u_values = []
    import dolfinx.geometry
    mesh = u.function_space.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, 2)
    cells = []
    points = []
    for i in range(len(points_xxyy[0])):
        point = [points_xxyy[0][i], points_xxyy[1][i], 0]
        #cell_candidates = dolfinx.geometry.compute_collisions_point(bb_tree, point)
        #cell = dolfinx.geometry.select_colliding_cells(mesh, cell_candidates, point, 1)
        cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, point)
        cell = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point)
        cells.append(cell[0])
        points.append(point)
    #u_values = u.eval(points_xy, cells)
    u_values = u.eval(points, cells)
    return u_values.T[0]


def myevalonline(u, xy1, xy2, numpoints=101):
    tol = 0.001
    xx = np.linspace(xy1[0], xy2[0], numpoints+0)
    yy = np.linspace(xy1[1], xy2[1], numpoints+0)
    points_xy = np.zeros((3, numpoints))
    points_xy[0] = xx
    points_xy[1] = yy
    points_xy=points_xy.T
    u_values = [] 
    import dolfinx.geometry
    mesh = u.function_space.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, 2)
    cells = []
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points_xy)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points_xy)
    for i, point in enumerate(points_xy):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    u_values = u.eval(points_on_proc, cells)
    return (points_xy.T,  u_values.T[0])

def myevalonline_grad_1(ugrad, V, xy1, xy2, numpoints=101):
    uu = project(ugrad, V)
    return myevalonline(uu.sub(0), xy1, xy2, numpoints=numpoints)

def myevalonline_grad_2(ugrad, V, V_grad, xy1, xy2, numpoints=101):
    ugrad_Expression = Expression(ugrad, V_grad.element.interpolation_points)
    ugrad_h = Function(V_grad)
    ugrad_h.interpolate(ugrad_Expression)
    uu = project(ugrad_h, V)
    #uu = project(ugrad, V)
    return myevalonline(uu.sub(0), xy1, xy2, numpoints=numpoints)


def mesh2triang(mesh):
    import matplotlib.tri as tri
    xy = mesh.geometry.x
    cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
    return tri.Triangulation(xy[:, 0], xy[:, 1], cells)

def myplot_2d(obj, ax=None, num_contours=10, plotmesh=True, aspect_equal=True, shading='gouraud', savename=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
        ax.cla()
    #plt.gca().set_aspect('equal')
    if aspect_equal == True:
        ax.set_aspect('equal')
        #ax.set_aspect('equal', 'box')
    if isinstance(obj, Mesh):
        #if (obj.geometry.dim != 2):
        #    raise(AttributeError)
        if plotmesh == True:
            ax.triplot(mesh2triang(obj), color='k')
    #if isinstance(obj, Function):
    else:
        mesh = obj.function_space.mesh
        mt = mesh2triang(mesh)
        C = obj.vector.array
        #print(len(C), len(mesh2triang(mesh).triangles))
        ax.tripcolor(mt, C, shading=shading) # shading= 'flat' alebo 'gouraud'
        if plotmesh == True:
            ax.triplot(mt, color='k')
        try:
            CS = ax.tricontour(mt, C, num_contours,linewidths=.5, colors='k')
            ax.clabel(CS, inline=1)
        except:
            pass
    if savename:
        plt.savefig(savename)

def myplot2D_(obj):
    fig, ax = plt.subplots(1, 1)
    ax.tripcolor(mesh2triang(obj.function_space.mesh), obj.vector.array)
    ax.triplot(mesh2triang(obj.function_space.mesh), color='k')
    plt.savefig("aa.png")

def myplot2Dmoj(obj, ax=None, num_contours=10, plotcolor=True, plotmesh=False, plotcontour=True, clabels=True, aspect_equal=True, shading='gouraud', savename=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ax.cla()
    #plt.gca().set_aspect('equal')
    if aspect_equal:
        ax.set_aspect('equal')
        #ax.set_aspect('equal', 'box')
    mesh = obj.function_space.mesh
    mt = mesh2triang(mesh)
    #C = obj.vector.array
    points = [[mt.x[i],mt.y[i]] for i in range(len(mt.x))]
    C =myevalonpoints(obj, points)
    if plotcolor:
        ax.tripcolor(mt, C, shading=shading) # shading= 'flat' alebo 'gouraud'
    if plotmesh == True:
        ax.triplot(mt, color='k')
    if plotcontour:
        try:
            CS = ax.tricontour(mt, C, num_contours,linewidths=1, colors='k')
            if clabels:
                ax.clabel(CS, inline=1)
        except:
            pass
    if savename:
        plt.savefig(savename)
        #plt.savefig("aa.png")
    return ax

####################################################################
### !!! toto zmenit na plotovanie priamo hodnot subdomain_tagov !!!
####################################################################
def pltsubdomains(obj, levels=None, ax=None, plotcolor=False, plotmesh=False, plotcontour=True, clabels=False, aspect_equal=True, shading='gouraud', savename=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ax.cla()
    #plt.gca().set_aspect('equal')
    if aspect_equal:
        #ax.set_aspect('equal')
        ax.set_aspect('equal', 'box')
    mesh = obj.function_space.mesh
    mt = mesh2triang(mesh)
    #C = obj.vector.array
    points = [[mt.x[i],mt.y[i]] for i in range(len(mt.x))]
    C =myevalonpoints(obj, points)
    if plotcolor:
        ax.tripcolor(mt, C, shading=shading) # shading= 'flat' alebo 'gouraud'
    if plotmesh == True:
        ax.triplot(mt, color='k')
    if plotcontour:
        levels = np.sort(levels)*.999
        try:
            CS = ax.tricontour(mt, C,levels=levels,linewidths=1, colors='k')
            if clabels:
                ax.clabel(CS, inline=1)
        except:
            pass
    if savename:
        plt.savefig(savename)
        #plt.savefig("aa.png")
    return ax
def myplot2Dcontour_(obj, plotmesh=False):
    fig, ax = plt.subplots(1, 1)
    mesh = obj.function_space.mesh
    C = obj.vector.array
    ax.tripcolor(mesh2triang(mesh), C)
    #ax.triplot(mesh2triang(obj.function_space.mesh), color='k')
    #try:
    if True:
        CS = ax.tricontour(mesh2triang(mesh), C, 10,linewidths=.5, colors='k')
        ax.clabel(CS, inline=1)
    #except:
    #    pass
    plt.savefig("aa.png")

#def mystreamplot_init(uu_projVV, numpoints_x=11, numpoints_y=11):
def mystreamplot(uu_projVV, ax=None, savename=None, color="k", numpoints_x=11, numpoints_y=11, aspect_equal=True):
    #uu_projVV je projekcia nejakeho gradientu do vektoroveho VV
    mesh = uu_projVV.function_space.mesh
    print("mesh")
    xmax = max(mesh.geometry.x[:,0])
    ymax = max(mesh.geometry.x[:,1])
    print("xmax ymax hotovo")
    xx=np.linspace(0, xmax, numpoints_x)
    yy=np.linspace(0, ymax, numpoints_y)
    print("xx yy hotovo")
    #xv, yv = np.meshgrid(xx, yy)
    vvx=np.array([[myeval(uu_projVV.sub(0),X,Y) for X in xx] for Y in yy])
    print("myeval x hotovo")
    vvy=np.array([[myeval(uu_projVV.sub(1),X,Y) for X in xx] for Y in yy])
    print("myeval y hotovo")
    if not ax:
        fig5 = plt.figure()
        ax = fig5.add_subplot(111)
    if aspect_equal:
        #ax.set_aspect('equal')
        ax.set_aspect('equal', 'box')
    ax.cla()
    #ax.streamplot(xx, yy, vvx, vvy, density=2, color="k")
    ax.streamplot(xx, yy, vvx, vvy, density=.5, color="k", broken_streamlines=False)
    print("streamplot hotovo")
    if savename:
        ax.figure.savefig(savename)
    
    return ax, xx,yy,vvx,vvy


def project(uu, V):
    u_, v_ = TrialFunction(V), TestFunction(V)
    a_p = inner(u_, v_) * dx
    L_p = inner(uu, v_) * dx
    projection = LinearProblem(a_p, L_p)
    return projection.solve()

