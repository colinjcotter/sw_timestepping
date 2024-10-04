from sw_setup import *

Unp1 = fd.Function(W)
u1, h1 = fd.split(Unp1)

"Crank-Nicholson rule"
half = fd.Constant(0.5)

u0, h0 = fd.split(Un)
eqn = (
    fd.inner(v, u1 - u0)*dx
    + half*dT*u_op(v, u0, h0)
    + half*dT*u_op(v, u1, h1)
    + phi*(h1 - h0)*dx
    + half*dT*h_op(phi, u0, h0)
    + half*dT*h_op(phi, u1, h1)
)
nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob, options_prefix="swe",
                                        solver_parameters=sparameters)
nsolver.set_transfer_manager(transfermanager)

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
nsteps = tcheck(tmax, dt)

for step in range(nsteps):
    PETSc.Sys.Print(f"\nTimestep {step} at time {t}, {t/tmax} of total\n")
    t += dt
    tdump += dt

    with PETSc.Log.Event("time solver"):
        nsolver.solve()
    Un.assign(Unp1)

    if args.one_step:
        step = nsteps-1

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/nsteps,
                "dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)
assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"

etan.assign(h0 - H + b)
un.assign(u0)
checkpoint_output(un, etan)
