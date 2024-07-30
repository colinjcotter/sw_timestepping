from sw_setup import *

Un_star = fd.Function(W)
Unp1 = fd.Function(W)
u1, h1 = fd.split(Unp1)

half = fd.Constant(0.5)
quarter = fd.Constant(0.25)

u0, h0 = fd.split(Un_star)

# a half step forward Euler
explicit_half = (
    fd.inner(v, u1 - u0)*dx
    + half*dT*u_op(v, u0, h0, system="linear")
    + phi*(h1 - h0)*dx
    + half*dT*h_op(phi, u0, h0, system="linear")
)

# a half step backward Euler
implicit_half = (
    fd.inner(v, u1 - u0)*dx
    + half*dT*u_op(v, u1, h1, system="linear")
    + phi*(h1 - h0)*dx
    + half*dT*h_op(phi, u1, h1, system="linear")
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

improb = fd.NonlinearVariationalProblem(implicit_half, Unp1)
imsolver = fd.NonlinearVariationalSolver(improb, options_prefix="linear_im",
                                        solver_parameters=lparams)

expprob = fd.NonlinearVariationalProblem(explicit_half, Unp1)
expsolver = fd.NonlinearVariationalSolver(expprob, options_prefix="linear_exp",
                                          solver_parameters=massparams)

theta = fd.Constant(0.55)
uh = (theta*u1 + (1-theta)*u0)/2
hh = (theta*h1 + (1-theta)*h0)/2
ubar = fd.Function(V1)

nonlinear = (
    fd.inner(v, u1 - u0)*dx
    + dT*u_op(v, uh, hh, system="nonlinear", vector_invariant=False, ubar=ubar)
    + phi*(h1 - h0)*dx
    + dT*h_op(phi, uh, hh, system="nonlinear", ubar=ubar)
)

nonlinearparams = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "fieldsplit_0": mass,
    "fieldsplit_1": mass
}

nonlinearprob = fd.NonlinearVariationalProblem(nonlinear, Unp1)
nonlinearsolver = fd.NonlinearVariationalSolver(expprob, options_prefix="nonlinear",
                                                solver_parameters=nonlinearparams)

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
un.assign(u0)
qsolver.solve()
fd.assemble(Courant_num_form, tensor=Courant_num)
Courant.interpolate(Courant_num/Courant_denom)

file_sw.write(un, etan, qn, Courant)

itcount = 0
stepcount = 0

ubar_scheme = "iterated"

while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {stepcount} at time {t}\n")
    t += dt
    tdump += dt

    if ubar_scheme == "half":
        # preliminary half timestep to get half value for ubar
        # half an advection step
        ubar.assign(0.5*u0)
        Un_star.assign(Un)
        with PETSc.Log.Event("advection step"):
            nonlinearsolver.solve()
        # half an implicit step
        Un_star.assign(Unp1)
        with PETSc.Log.Event("implicit solver"):
            imsolver.solve()

        # full timestep using that ubar value
        ubar.assign(u1)
        # half an explicit step
        Un_star.assign(Un)
        with PETSc.Log.Event("explicit solver"):
            expsolver.solve()
        # a full advection step
        Un_star.assign(Unp1)
        with PETSc.Log.Event("advection step"):
            nonlinearsolver.solve()
        # half an implicit step
        Un_star.assign(Unp1)
        with PETSc.Log.Event("implicit solver"):
            imsolver.solve()
        Un.assign(Unp1)
    elif ubar_scheme == "iterated":
        ubar.assign(u0)
        for i in range(4):
            # half an explicit step
            Un_star.assign(Un)
            with PETSc.Log.Event("explicit solver"):
                expsolver.solve()
            # a full advection step
            Un_star.assign(Unp1)
            with PETSc.Log.Event("advection step"):
                nonlinearsolver.solve()
            # half an implicit step
            Un_star.assign(Unp1)
            with PETSc.Log.Event("implicit solver"):
                imsolver.solve()
            ubar.assign((u0 + u1)/2)
        Un.assign(Unp1)

    if args.one_step:
        t = tmax + dt

    print('Energy: ',fd.assemble(0.5*h0*fd.inner(u0, u0)*fd.dx + 0.5*g*(h0-H+b)**2*fd.dx))
    fd.assemble(Courant_num_form, tensor=Courant_num)
    Courant.interpolate(Courant_num/Courant_denom)
    print(Courant.dat.data[:].max())
    
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

