from sw_setup import *
from irksome import Dt, MeshConstant, RadauIIA, TimeStepper
MC = MeshConstant(mesh)

dT = MC.Constant(dt)
t = MC.Constant(0.)

butcher_tableau = RadauIIA(2)

u0, h0 = fd.split(Un)
eqn = (
    fd.inner(v, Dt(u0))*dx
    + u_op(v, u0, h0)
    + phi*(Dt(h0))*dx
    + h_op(phi, u0, h0)
)

stepper = TimeStepper(eqn, butcher_tableau, t, dT, Un,
                      solver_parameters=sparameters)
stepper.solver.set_transfer_manager(transfermanager)

dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

from firedrake.output import VTKFile
file_sw = VTKFile(name+'.pvd')
u0, h0 = Un.subfunctions
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)

itcount = 0
stepcount = 0
while float(t) < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {stepcount} at time {float(t)}, Tmax {tmax}\n")
    t.assign(float(t) + dt)
    tdump += dt

    stepper.advance()

    if args.one_step:
        t.assign(tmax + dt)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    stepcount += 1
