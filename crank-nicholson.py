from sw_setup import *

Unp1 = fd.Function(W)
if args.hybrid:
    u1, h1, ll1 = fd.split(Unp1)
else:
    u1, h1 = fd.split(Unp1)

"Crank-Nicholson rule"
half = fd.Constant(0.5)

if args.hybrid:
    u0, h0, ll0 = fd.split(Un)
else:
    u0, h0 = fd.split(Un)

eqn = (
    fd.inner(v, u1 - u0)*dx
    + half*dT*u_op(v, u0, h0)
    + half*dT*u_op(v, u1, h1)
    + phi*(h1 - h0)*dx
    + half*dT*h_op(phi, u0, h0)
    + half*dT*h_op(phi, u1, h1)
)

if args.hybrid:
    eqn += dT*fd.inner(fd.jump(v, n), ll1)*fd.dS
    eqn += fd.inner(fd.jump(u1, n), mu)*fd.dS

if args.hybrid:
    dim = 1
else:
    dim = 0
    
pc = {
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_pc_patch_construct_dim": dim,
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_local_type": "additive",
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_precompute_element_tensors": True,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu"
}

nomgparameters = {
    "snes_monitor": None,
    "snes_lag_preconditioner": 10,
    "snes_lag_preconditioner_persists": None,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    #"ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-50,
    "ksp_rtol": 1e-5,
    "ksp_max_it": 400,
    "pc_type": "ksp",
    "ksp_ksp_type": "richardson",
    "ksp_ksp_richardson_scale": 1.,
    "ksp_ksp_rtol": 1e-10,
    "ksp_ksp_max_it": 3,
    "ksp_ksp_convergence_test": 'skip',
    "ksp_ksp_converged_maxits": None,
    "ksp" : pc
}

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob, options_prefix="swe",
                                        solver_parameters=nomgparameters)
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

u0, h0 = swap(Un.subfunctions)

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
