from sw_setup import *

Unp1 = fd.Function(W)
u1, h1 = fd.split(Unp1)

half = fd.Constant(0.5)
quarter = fd.Constant(0.25)

u0, h0 = fd.split(Un)

energy_expr = 0.5*(fd.inner(u0, u0)*h0 + g*(h0**2 + h0*b))*fd.dx

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
file_sw.write(un, etan, qn)

itcount = 0
stepcount = 0
energy0 = fd.assemble(energy_expr)
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {stepcount} at time {t} of {tmax}\n")
    t += dt
    tdump += dt

    # half an implicit step
    with PETSc.Log.Event("implicit solver"):
        imsolver.solve()
        Un.assign(Unp1)

    # a full explicit step using
    # heun scheme
    # yhat = y_0 + hf(y_0)
    # y_1 = y_0 + h/2*(f(y_0) + f(yhat))
    #     = (y_0 + h*f(y_0))/2 + (y_0 + h*f(yhat))/2)
    with PETSc.Log.Event("explicit solver"):
        Uhat.assign(Un)
        expsolver.solve()  #  Unp1 contains Un + h*f(y_0) = yhat
        Uhat.assign(Unp1) #  Uhat contains yhat
        expsolver.solve()  #  Unp1 contains Un + h*f(yhat)
        Un.assign( (Uhat + Unp1)/2 )

    # half an implicit step
    with PETSc.Log.Event("implicit solver"):
        imsolver.solve()
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
