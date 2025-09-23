from sw_setup import *

u0 = Un.sub(0)
G0 = Un.sub(1)
Unp1 = fd.Function(W)
u1 = Unp1.sub(0)
G1 = Unp1.sub(1)
uh = (u0 + u1)/2
Gh = (G0 + G1)/2

dT = fd.Constant(dt)

eqn = (
    fd.inner(v, u1-u0)*dx
    + dT*u_op(v, uh, -fd.div(Gh))
    + fd.inner(dG, G1-G0)*dx
    + dT*G_op(dG, uh, Gh)
)

fd.derivative(eqn, Unp1)


Prob0 = fd.NonlinearVariationalProblem(eqn, Unp1)
Solver0 = fd.NonlinearVariationalSolver(Prob0, solver_parameters=lu)

tdump = 0.
tn = 0.

from firedrake.output import VTKFile
file_sw = VTKFile(name+'.pvd')
if args.vfs:
    U0, G0 = Un.subfunctions
    etan.interpolate(-fd.div(G0))
else:
    u0, h0 = Un.subfunctions
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)

nsteps = tcheck(tmax, dt)
step = 0
t = 0.

for step in range(nsteps):
    PETSc.Sys.Print(f"\nTimestep {step} of {nsteps}. dt={dt}\n")

    tdump += dt
    t += dt

    Solver0.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt

    if args.one_step:
        t = tmax
        break
    
assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"
