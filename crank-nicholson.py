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

hdump1 = args.checkpnt_dumpt
dumpt1 = hdump1*60.*60.
tdump1 = 0.



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
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {stepcount} at time {t}\n")
    t += dt
    tdump += dt
    tdump1 += dt

    with PETSc.Log.Event("time solver"):
        nsolver.solve()
    Un.assign(Unp1)

    if args.one_step:
        t = tmax + dt

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt

    PETSc.Sys.Print('Energy: ',fd.assemble(0.5*h0*fd.inner(u0, u0)*fd.dx + 0.5*g*(h0-H+b)**2*fd.dx))
        
    if tdump1 > dumpt1 - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        with fd.CheckpointFile("example2.h5", 'w') as afile:
            afile.save_mesh(mesh)  # optional
            afile.save_function(un)
            afile.save_function(etan)
            afile.save_function(qn)
        print('\nCheckpointing!!!')
        tdump1 -= dumpt1

    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount,
                "dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)

