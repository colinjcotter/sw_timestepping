import firedrake as fd
#get command arguments
from petsc4py import PETSc
from firedrake.__future__ import interpolate
from irksome import Dt, MeshConstant, IRKAuxiliaryOperatorPC
from irksome.pc import RanaLD

import numpy as np
import mg
import argparse

parser = argparse.ArgumentParser(description='Williamson 5 testcase.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--tmax', type=float, default=1296000, help='Final time in seconds. Default 1296000 (15 days).')
parser.add_argument('--dumpt', type=float, default=86400, help='Dump time in seconds. Default 86400 (24 hours).')
parser.add_argument('--dt', type=float, default=3600, help='Timestep in seconds. Default 1.')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--one_step', action='store_true', help='Do one timestep and exit (overriding dmax).')
parser.add_argument('--filename', type=str, default='w5')
parser.add_argument('--checkpointfile', type=str, default='none')
parser.add_argument('--vector_invariant', action='store_true', help='Use the vector invariant form.')
parser.add_argument('--bdfm', action='store_true', help='Use the BDFM space.')
parser.add_argument('--hybrid', action='store_true', help='Use broken formulation with trace multipliers.')
parser.add_argument('--rk_stages', type=int, default=2, help='Number of RK stages in IRK.')
parser.add_argument('--rk_type', type=str, default='RadauIIA', help='RadauIIA or GaussLegendre')
parser.add_argument('--sdc', action='store_true', help='Use SDC preconditioner in IRK.')
parser.add_argument('--centred', action='store_true', help='Use centred fluxes.')
parser.add_argument('--ntol', type=float, default=1.0e-8, help='Solver tolerance for the nonlinear solver')
parser.add_argument('--ktol', type=float, default=1.0e-10, help='Solver tolerance for the linear solver')
parser.add_argument('--gamma', type=float, default=0.0, help='Augmented Lagrangian parameter.')
parser.add_argument('--williamson', type=int, default=5, help='Williamson testcase number.')
parser.add_argument('--pcscheme', type=str, default="mono", help='Preconditioner option: mono - monolithic patch PC (default),  mg - mg with monolithic patch PC, rana - rana block preconditioner with mg on the blocks, waverana - rana block preconditioner using linearisation about state of rest. al - monolithic augmented Lagrangian')

args = parser.parse_known_args()
args = args[0]

vector_invariant = args.vector_invariant

tmax = args.tmax
dumpt = args.dumpt

if args.show_args:
    PETSc.Sys.Print(args)
    
# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
base_level = args.base_level
nrefs = args.ref_level - base_level
name = args.filename
deg = args.coords_degree
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

def high_order_mesh_hierarchy(mh, degree, R0):
    meshes = []
    for m in mh:
        X = fd.VectorFunctionSpace(m, "Lagrange", degree)
        new_coords = fd.Function(X).interpolate(m.coordinates)
        x, y, z = new_coords
        r = (x**2 + y**2 + z**2)**0.5
        new_coords = fd.Function(X).interpolate(R0*new_coords/r)
        new_mesh = fd.Mesh(new_coords, name="errormesh")
        meshes.append(new_mesh)

    return fd.HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)

basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=base_level,
                                    #degree=args.coords_degree,
                                    distribution_parameters = distribution_parameters)
del basemesh._radius
mh = fd.MeshHierarchy(basemesh, nrefs)
mh = high_order_mesh_hierarchy(mh, deg, R0)
for mesh in mh:
    xf = mesh.coordinates
    mesh.transfer_coordinates = fd.Function(xf)
    x = fd.SpatialCoordinate(mesh)
    r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
    xf.interpolate(R0*xf/r)
    mesh.init_cell_orientations(x)
mesh = mh[-1]
mesh.name="errormesh"

R0 = fd.Constant(R0)
x = fd.SpatialCoordinate(mesh)
cx, cy, cz = x

outward_normals = fd.CellNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
if args.bdfm:
    family = "BDFM"
else:
    family = "BDM"

if args.hybrid:
    V1_ele = fd.FiniteElement(family, fd.triangle, degree+1)
    V1 = fd.FunctionSpace(mesh, fd.BrokenElement(V1_ele))
else:
    V1 = fd.FunctionSpace(mesh, family, degree+1)
V1dg = fd.VectorFunctionSpace(mesh, "DG", degree+1, dim=3)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
if args.hybrid:
    if args.bdfm:
        T = fd.FunctionSpace(mesh, "HDivT", degree)
    else:
        T = fd.FunctionSpace(mesh, "HDivT", degree+1)
    W = fd.MixedFunctionSpace((V1, V2, T))
else:
    W = fd.MixedFunctionSpace((V1, V2))

if args.hybrid:
    u, eta, ll = fd.TrialFunctions(W)
    v, phi, mu = fd.TestFunctions(W)
else:
    u, eta = fd.TrialFunctions(W)
    v, phi = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)

# D = eta + b

One = fd.Function(V2).assign(1.0)

dx = fd.dx

Un = fd.Function(W)
if args.hybrid:
    u0, h0, ll0 = fd.split(Un)
else:
    u0, h0 = fd.split(Un)
n = fd.FacetNormal(mesh)

def both(u):
    return 2*fd.avg(u)

dT = fd.Constant(0.)
dS = fd.dS

def u_op(v, u, h, system="full"):
    if args.centred:
        Upwind = 0.5
    else:
        Upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
    K = 0.5*fd.inner(u, u)
    if vector_invariant:
        nonlinear = ( - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*dx
                      + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                                 both(Upwind*u))*dS
                      - fd.div(v)*K*dx)
    else:
        nonlinear = -fd.inner(fd.div(fd.outer(v, u)), u)*fd.dx
        if args.centred:
            unp = 0.5*fd.dot(u('+'), n('+'))
            unm = 0.5*fd.dot(u('-'), n('-'))
        else:
            unp = \
                0.5*(fd.dot(u('+'), n('+')) + abs(fd.dot(u('+'), n('+'))))
            unm = \
                0.5*(fd.dot(u('-'), n('-')) + abs(fd.dot(u('-'), n('-'))))
        nonlinear += fd.dot(fd.jump(v),
                            (unp*u('+') - unm*u('-')))*dS
    linear = fd.inner(v, f*perp(u))*dx - fd.div(v)*g*(h+b)*dx
    if system == "linear":
        return linear
    if system == "nonlinear":
        return nonlinear
    return linear + nonlinear

def h_op(phi, u, h, system="full"):
    if system == "linear":
        return H*fd.div(u)*phi*dx
    if args.centred:
        unp = 0.5*fd.dot(u('+'), n('+'))
        unm = 0.5*fd.dot(u('-'), n('-'))
    else:
        unp = 0.5*(fd.dot(u('+'), n('+')) + abs(fd.dot(u('+'), n('+'))))
        unm = 0.5*(fd.dot(u('-'), n('-')) + abs(fd.dot(u('-'), n('-'))))
    if system == "nonlinear":
        return (- fd.inner(fd.grad(phi), u)*(h-H)*dx
                + fd.jump(phi)*(unp*(h('+')-H)
                                - unm*(h('-')-H))*dS
                )
    return (- fd.inner(fd.grad(phi), u)*h*dx
            + fd.jump(phi)*(unp*h('+') - unm*h('-'))*dS
            )

# monolithic solver options

mgparameters = {
    "snes_monitor": None,
    "snes_ksp_ew": None,
    "snes_atol": 0.,
    "snes_stol": 0.,
    "snes_rtol": args.ntol,
    "snes_linesearch_type": "basic",
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    #"ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_atol": 0.,
    "ksp_rtol": args.ktol,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_ksp_convergence_test": "skip",
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": True,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_pc_patch_construct_type": "star",
    "mg_levels_patch_pc_patch_local_type": "additive",
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "ilu",
    "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",

    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.PatchPC",
    "mg_coarse_patch_pc_patch_save_operators": True,
    "mg_coarse_patch_pc_patch_partition_of_unity": True,
    "mg_coarse_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_coarse_patch_pc_patch_construct_dim": 0,
    "mg_coarse_patch_pc_patch_construct_type": "star",
    "mg_coarse_patch_pc_patch_local_type": "additive",
    "mg_coarse_patch_pc_patch_precompute_element_tensors": True,
    "mg_coarse_patch_pc_patch_symmetrise_sweep": False,
    "mg_coarse_patch_sub_ksp_type": "preonly",
    "mg_coarse_patch_sub_pc_type": "ilu",
    "mg_coarse_patch_sub_pc_factor_shift_type": "nonzero",
    
    #"mg_coarse_pc_type": "python",
    #"mg_coarse_pc_python_type": "firedrake.AssembledPC",
    #"mg_coarse_assembled_pc_type": "lu",
    #"mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
}

starasm = {
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_sub_sub_pc_type": "ilu",
    "assembled_pc_star_sub_sub_ksp_type": "preonly",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_backend": "tinyasm",
}

patch0 = {
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_pc_patch_construct_dim": 0,
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_local_type": "additive",
    #"patch_pc_patch_precompute_element_tensors": True,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
    "patch_sub_pc_factor_shift_type": "nonzero"
}

patch = {
    "pc_type": "python"
    } | patch0

mgopts = {
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 1,
    "mg_levels_ksp_convergence_test": "skip",
    "mg_levels" : patch,
    "mg_coarse": patch,
}

lu = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
    }

ilu = {
    'ksp_type': 'gmres',
    'ksp_max_it': 3,
    'pc_type': 'bjacobi',
    'sub_pc_type': 'ilu'
    }
    
vtransfer = mg.ManifoldTransfer()
tm = fd.TransferManager()
transfers = {
    V1.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
                       vtransfer.inject),
    V2.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
                       vtransfer.inject)
}
transfermanager = fd.TransferManager(native_transfers=transfers)

MC = MeshConstant(mesh)
dt = args.dt
dT = MC.Constant(dt)
tc = MC.Constant(0.)
gamma = MC.Constant(args.gamma)

x = fd.SpatialCoordinate(mesh)
un = fd.Function(V1, name="Velocity")
etan = fd.Function(V2, name="Elevation")

if args.hybrid:
    u0, h0, ll0 = Un.subfunctions
else:
    u0, h0 = Un.subfunctions

testcase = args.williamson

if testcase == 5:
    u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
    u_max = fd.Constant(u_0)
    u_expr = fd.as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
    un.project(u_expr)
    etan.project(eta_expr)
    # Topography.
    rl = fd.pi/9.0
    lambda_x = fd.atan2(x[1]/R0, x[0]/R0)
    lambda_c = -fd.pi/2.0
    phi_x = fd.asin(x[2]/R0)
    phi_c = fd.pi/6.0
    minarg = fd.min_value(pow(rl, 2),
                          pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
    bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
    b.interpolate(bexpr)
    u0.assign(un)
    h0.assign(etan + H - b)

elif testcase == 6:
    x, y, z = fd.SpatialCoordinate(mesh)
    lon = fd.atan2(y, x)
    l = (x**2 + y**2)**0.5
    lat = fd.atan2(z, l)

    # code stolen from Alex Brown
    R = fd.Constant(4)
    K = fd.Constant(7.847e-6) # Frequency parameter, in sec^-1
    w = K
    H0 = fd.Constant(8000.)
    psi = fd.Function(V0)
    psiexpr = -R0**2 * w * fd.sin(lat) + \
        R0**2 * K * fd.cos(lat)**R * fd.sin(lat) * fd.cos(R*lon)
    psi.interpolate(psiexpr)
    u_expr = perp(fd.grad(psi))
    un.project(u_expr)
    u0.assign(un)
    # Initilising the depth field
    A = (w / 2) * (2 * Omega + w) * fd.cos(lat)**2 + \
        0.25 * K**2 * fd.cos(lat)**(2 * R) * ((R + 1) * fd.cos(lat)**2 + (2 * R**2 - R - 2) - 2 * R**2 * fd.cos(lat)**(-2))
    B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
    B = B_frac * fd.cos(lat)**R * ((R**2 + 2 * R + 2) - (R + 1)**2 * fd.cos(lat)**2)
    C = (1 / 4) * K**2 * fd.cos(lat)**(2 * R) * ((R + 1)*fd.cos(lat)**2 - (R + 2))
    Dexpr = H0 + R0**2 * (A + B*fd.cos(lon*R) + C * fd.cos(2 * R * lon))/g
    h0.interpolate(Dexpr)
else:
    raise NotImplementedError

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'preonly',
           'pc_type':'lu',
           "pc_factor_mat_solver_type": "superlu_dist"}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

def checkpoint_output(u_out, eta_out):
    if args.checkpointfile == 'none':
        return

    with fd.CheckpointFile(args.checkpointfile, 'w') as afile:
        u_out_dg = fd.Function(V1dg, name="Velocity")
        fd.assemble(interpolate(u_out, V1dg), tensor=u_out_dg)
        afile.save_function(u_out_dg)
        afile.save_function(eta_out)

def tcheck(tmax, dt):
    nsteps = round(tmax/dt)
    from math import fabs
    assert fabs(nsteps*dt - tmax) < 1.0e-5, "tmax is not integer multiple of dt"
    return nsteps
