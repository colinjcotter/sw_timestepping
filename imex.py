from sw_setup import *
#  ARK3(2,2,2) scheme from Giraldo et al (2013)

Unp1 = fd.Function(W)
u1, h1 = fd.split(Unp1)

half = fd.Constant(0.5)
quarter = fd.Constant(0.25)

u0, h0 = fd.split(Un)

energy_expr = 0.5*(fd.inner(u0, u0)*h0 + g*(h0**2 + h0*b))*fd.dx

lparams = {
    "mat_type": "matfree",
    "snes_lag_jacobian": -2,
    "snes_lag_jacobian_persists": None,
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "python",
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {'ksp_type': 'preonly',
                      'pc_type': 'lu',
                      "pc_factor_mat_solver_type":'superlu_dist'
                      }}

# stage functions
Uk2 = fd.Function(W)
Uk3 = fd.Function(W)

# some coefficients
gamma = fd.Constant(1. - 0.5**0.5)
alpha = fd.Constant((3 + 2*2.**0.5)/6)
delta = fd.Constant(0.5*0.5**0.5)

uk1, hk1 = fd.split(Un)  #  Uk1 is just Un for ARK2(2,3,2)
uk2, hk2 = fd.split(Uk2)  #  Uk1 is just Un for ARK2(2,3,2)
uk3, hk3 = fd.split(Uk3)  #  Uk1 is just Un for ARK2(2,3,2)

k2_eqn = (
    fd.inner(v, uk2 - u0)*dx
    + 2*gamma*dT*u_op(v, uk1, hk1, system="nonlinear")
    + gamma*dT*u_op(v, uk1, hk1, system="linear")
    + gamma*dT*u_op(v, uk2, hk2, system="linear")
    + phi*(hk2 - h0)*dx
    + 2*gamma*dT*h_op(phi, uk1, hk1, system="nonlinear")
    + gamma*dT*h_op(phi, uk1, hk1, system="linear")
    + gamma*dT*h_op(phi, uk2, hk2, system="linear")
)

k2prob = fd.NonlinearVariationalProblem(k2_eqn, Uk2)
k2solver = fd.NonlinearVariationalSolver(k2prob, options_prefix="k2",
                                         solver_parameters=lparams)

k3_eqn = (
    fd.inner(v, uk3 - u0)*dx
    + (1-alpha)*dT*u_op(v, uk1, hk1, system="nonlinear")
    + alpha*dT*u_op(v, uk2, hk2, system="nonlinear")
    + delta*dT*u_op(v, uk1, hk1, system="linear")
    + delta*dT*u_op(v, uk2, hk2, system="linear")
    + gamma*dT*u_op(v, uk3, hk3, system="linear")
    + phi*(hk3 - h0)*dx
    + (1-alpha)*dT*h_op(phi, uk1, hk1, system="nonlinear")
    + alpha*dT*h_op(phi, uk2, hk2, system="nonlinear")
    + delta*dT*h_op(phi, uk1, hk1, system="linear")
    + delta*dT*h_op(phi, uk2, hk2, system="linear")
    + gamma*dT*h_op(phi, uk3, hk3, system="linear")
)

k3prob = fd.NonlinearVariationalProblem(k3_eqn, Uk3)
k3solver = fd.NonlinearVariationalSolver(k3prob, options_prefix="k3",
                                         solver_parameters=lparams)

mass = {
    "ksp_type": "gmres",
    "pc_type": "bjacobi",
    "sub_pc_type": "ilu"
}

massparams = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "fieldsplit_0": mass,
    "fieldsplit_1": mass
}

unp1_eqn = (
    fd.inner(v, u1 - u0)*dx
    + delta*dT*u_op(v, uk1, hk1, system="nonlinear")
    + delta*dT*u_op(v, uk2, hk2, system="nonlinear")
    + gamma*dT*u_op(v, uk3, hk3, system="nonlinear")
    + delta*dT*u_op(v, uk1, hk1, system="linear")
    + delta*dT*u_op(v, uk2, hk2, system="linear")
    + gamma*dT*u_op(v, uk3, hk3, system="linear")
    + phi*(h1 - h0)*dx
    + delta*dT*h_op(phi, uk1, hk1, system="nonlinear")
    + delta*dT*h_op(phi, uk2, hk2, system="nonlinear")
    + gamma*dT*h_op(phi, uk3, hk3, system="nonlinear")
    + delta*dT*h_op(phi, uk1, hk1, system="linear")
    + delta*dT*h_op(phi, uk2, hk2, system="linear")
    + gamma*dT*h_op(phi, uk3, hk3, system="linear")
)

unp1prob = fd.NonlinearVariationalProblem(unp1_eqn, Unp1)
unp1solver = fd.NonlinearVariationalSolver(unp1prob, options_prefix="unp1",
                                           solver_parameters=massparams)

Unp1.assign(Un)

dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.
t = 0.
PETSc.Sys.Print('tmax', tmax, 'dt', dt)

u0, h0 = Un.subfunctions

from firedrake.output import VTKFile
file_sw = VTKFile(name+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)

itcount = 0
stepcount = 0
energy0 = fd.assemble(energy_expr)
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {stepcount} at time {t}, {t/tmax} of total\n")
    t += dt
    tdump += dt

    k2solver.solve()
    k3solver.solve()
    unp1solver.solve()
    Un.assign(Unp1)

    if args.one_step:
        t = tmax + dt

    energy = fd.assemble(energy_expr)
    PETSc.Sys.Print("relative energy error", (energy-energy0)/energy0)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    stepcount += 1
PETSc.Sys.Print("dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)

etan.assign(h0 - H + b)
un.assign(u0)
checkpoint_output(un, etan)
