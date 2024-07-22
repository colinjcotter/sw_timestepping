from sw_setup import *

Unp1 = fd.Function(W)
u1, h1 = fd.split(Unp1)

half = fd.Constant(0.5)
quarter = fd.Constant(0.25)

u0, h0 = fd.split(Un)

linear = (
    fd.inner(v, u1 - u0)*dx
    + quarter*dT*u_op(v, u0, h0, system="linear")
    + quarter*dT*u_op(v, u1, h1, system="linear")
    + phi*(h1 - h0)*dx
    + quarter*dT*h_op(phi, u0, h0, system="linear")
    + quarter*dT*h_op(phi, u1, h1, system="linear")
)

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

improb = fd.NonlinearVariationalProblem(linear, Unp1)
imsolver = fd.NonlinearVariationalSolver(improb, options_prefix="linear",
                                        solver_parameters=lparams)

Uhat = fd.Function(W) # - working memory
uh, hh = fd.split(Uhat)

nonlinear = (
    fd.inner(v, u1 - u0)*dx
    + dT*u_op(v, uh, hh, system="nonlinear")
    + phi*(h1 - h0)*dx
    + dT*h_op(phi, uh, hh, system="nonlinear")
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
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "fieldsplit_0": mass,
    "fieldsplit_1": mass
}

expprob = fd.NonlinearVariationalProblem(nonlinear, Unp1)
expsolver = fd.NonlinearVariationalSolver(expprob, options_prefix="exp",
                                          solver_parameters=mass)

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
fd.assemble(Courant_num_form, tensor=Courant_num)
Courant.interpolate(Courant_num/Courant_denom)

file_sw.write(un, etan, qn, Courant)

itcount = 0
stepcount = 0
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {stepcount} at time {t}\n")
    t += dt
    tdump += dt

    # half an implicit step
    with PETSc.Log.Event("implicit solver"):
        imsolver.solve()
        Un.assign(Unp1)

    # a full explicit step using
    # SSPRK3
    with PETSc.Log.Event("explicit solver"):
        Uhat.assign(Un)
        expsolver.solve()  #  Unp1 contains U_1 = Un + h*f(Un)
        Uhat.assign(Unp1)
        expsolver.solve()  #  Unp1 contains U_2 = U_1 + h*f(U_1)
        Uhat.assign(.75*Uhat + 0.25*Unp1)  # U_2 -> 0.75*Un + 0.25*U_2
        expsolver.solve()  #  Unp1 contains U_3 = U_2 + h*f(U_2)
        Un.assign(Un/3 + 2*Unp1/3)

    # half an implicit step
    with PETSc.Log.Event("implicit solver"):
        imsolver.solve()
        Un.assign(Unp1)

    if args.one_step:
        t = tmax + dt

    print('Energy: ',fd.assemble(0.5*h0*fd.inner(u0, u0)*fd.dx + 0.5*g*(h0-H+b)**2*fd.dx))

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

