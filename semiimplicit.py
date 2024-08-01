from sw_setup import *

Un_in = fd.Function(W)
Un_out = fd.Function(W)
Unp1 = fd.Function(W)
delta_Unp1 =fd.Function(W)

half = fd.Constant(0.5)
quarter = fd.Constant(0.25)

u0, h0 = fd.split(Un_in)
u1, h1 = fd.split(Un_out)

# as per the book, solve
# L \Delta U = U^* - U^{n+1}
# where U^* is obtained by
# doing half a step of forcing using U^n
# then a full step of advection,
# then half a step of forcing using U^{n+1}
# L is the linearised operator about state of rest
# then update U^{n+1} -> U^{n+1} + \Delta U

# a half step forward Euler
explicit_half = (
    fd.inner(v, u1 - u0)*dx
    + half*dT*u_op(v, u0, h0, system="linear")
    + phi*(h1 - h0)*dx
    + half*dT*h_op(phi, u0, h0, system="linear")
)

# a full step advection
theta = fd.Constant(0.50)
uh = (theta*u1 + (1-theta)*u0)
hh = (theta*h1 + (1-theta)*h0)
un0, _ = fd.split(Un)
unp1, hnp1 = fd.split(Unp1)
ubar = theta*unp1 + (1-theta)*un0

nonlinear = (
    fd.inner(v, u1 - u0)*dx
    + dT*u_op(v, uh, hh, system="nonlinear",
              vector_invariant=False, ubar=ubar)
    + phi*(h1 - h0)*dx
    + dT*h_op(phi, uh, hh, system="nonlinear", ubar=ubar)
)

# a half step forward Euler using Unp1
explicit_half_Unp1 = (
    fd.inner(v, u1 - u0)*dx
    + half*dT*u_op(v, unp1, hnp1, system="linear")
    + phi*(h1 - h0)*dx
    + half*dT*h_op(phi, unp1, hnp1, system="linear")

)

# a linearised solve on the residuals
du, dh = fd.split(delta_Unp1)

implicit = (
    fd.inner(v, du + unp1 - u1)*dx
    + half*dT*u_op(v, du, dh, system="linear")
    + phi*(dh + hnp1 - h1)*dx
    + half*dT*h_op(phi, du, dh, system="linear")
)

mass = {
    "ksp_type": "gmres",
    "pc_type": "bjacobi",
    "sub_pc_type": "ilu"
}

massparams = {
    "snes_lag_jacobian": -2,
    "snes_lag_jacobian_persists": None,
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "fieldsplit_0": mass,
    "fieldsplit_1": mass
}

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

exp1prob = fd.NonlinearVariationalProblem(explicit_half, Un_out)
exp1solver = fd.NonlinearVariationalSolver(exp1prob,
                                           options_prefix="linear_exp",
                                           solver_parameters=massparams)
exp2prob = fd.NonlinearVariationalProblem(explicit_half_Unp1, Un_out)
exp2solver = fd.NonlinearVariationalSolver(exp2prob,
                                           options_prefix="linear_exp",
                                           solver_parameters=massparams)

nonlinearparams = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "fieldsplit_0": mass,
    "fieldsplit_1": mass
}

nonlinearprob = fd.NonlinearVariationalProblem(nonlinear, Un_out)
nonlinearsolver = fd.NonlinearVariationalSolver(
    nonlinearprob, options_prefix="nonlinear",
    solver_parameters=nonlinearparams)

linearprob = fd.NonlinearVariationalProblem(implicit, delta_Unp1)
linearsolver = fd.NonlinearVariationalSolver(
    linearprob, options_prefix="linear",
    solver_parameters=lparams)

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
u1, h1 = Unp1.subfunctions

from firedrake.output import VTKFile
file_sw = VTKFile(name+'.pvd')
etan.assign(h0 - H + b)
qsolver.solve()
fd.assemble(Courant_num_form, tensor=Courant_num)
Courant.interpolate(Courant_num/Courant_denom)

file_sw.write(un, etan, qn, Courant)

itcount = 0
stepcount = 0

while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {stepcount} at time {t}\n")
    t += dt
    tdump += dt

    Unp1.assign(Un)
    for i in range(args.siits):
        Un_in.assign(Un)
        exp1solver.solve()
        Un_in.assign(Un_out)
        nonlinearsolver.solve()
        Un_in.assign(Un_out)
        exp2solver.solve()
        linearsolver.solve()
        Unp1 += delta_Unp1
    Un.assign(Unp1)

    if args.one_step:
        t = tmax + dt

    PETSc.Sys.Print('Energy: ',fd.assemble(0.5*h0*fd.inner(u0, u0)*fd.dx + 0.5*g*(h0-H+b)**2*fd.dx))
    fd.assemble(Courant_num_form, tensor=Courant_num)
    Courant.interpolate(Courant_num/Courant_denom)
    PETSc.Sys.Print(Courant.dat.data[:].max())
    
    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        fd.assemble(Courant_num_form, tensor=Courant_num)
        Courant.interpolate(Courant_num/Courant_denom)

        file_sw.write(un, etan, qn, Courant)
        tdump -= dumpt
    stepcount += 1
PETSc.Sys.Print("dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)

