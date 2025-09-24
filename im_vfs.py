from sw_setup import *

u0 = Un[0, :]
G0 = Un[1, :]
Unp1 = fd.Function(W)
u1 = Unp1[0, :]
G1 = Unp1[1, :]
uh = (u0 + u1)/2
Gh = (G0 + G1)/2

dT = fd.Constant(dt)

eqn = (
    fd.inner(v, u1-u0)*dx
    + dT*u_op(v, uh, H-fd.div(Gh))
    + fd.inner(dG, G1-G0)*dx
    + dT*G_op(dG, uh, Gh)
)

Prob0 = fd.NonlinearVariationalProblem(eqn, Unp1)
luparams = {
    "snes_monitor": None,
    "snes_ksp_ew": None,
    "snes_lag_preconditioner": 10,
    "snes_lag_preconditioner_persists": None,
    "snes_stol": 0,
    "snes_atol": 0,
    "snes_rtol": args.ntol,
    "mat_type": "aij",
    "ksp_type": "gmres",
    "ksp_monitor": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
    }

Solver0 = fd.NonlinearVariationalSolver(Prob0, solver_parameters=luparams)

tdump = 0.
tn = 0.

from firedrake.output import VTKFile
file_sw = VTKFile('vfs.pvd')
un0 = Un[0, :]
Gn0 = Un[1, :]
etan.interpolate(-fd.div(Gn0))
un.interpolate(un0)
Gview = fd.Function(V1)
qsolver.solve()
Gview.interpolate(Gn0)
file_sw.write(un, etan, qn, Gview)

nsteps = tcheck(tmax, dt)
step = 0
t = 0.

Unp1.assign(Un)

for step in range(nsteps):
    PETSc.Sys.Print(f"\nTimestep {step} of {nsteps}. dt={dt}\n")

    tdump += dt
    t += dt

    Solver0.solve()
    #h0.project(fd.div(Unp1[1,:]))
    #hmean = fd.assemble(h0*dx)/fd.assemble(One*dx)
    #print("hmean", hmean)
    #h0.assign(h0-hmean)
    #G_setup_solver.solve()
    #Gview.interpolate(vg)
    #Unp1.sub(1).assign(Gview)
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        etan.interpolate(-fd.div(Gn0))
        un.interpolate(un0)
        qsolver.solve()
        Gview.interpolate(Gn0)
        file_sw.write(un, etan, qn, Gview)
        tdump -= dumpt

    if args.one_step:
        t = tmax
        break
    
assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"
