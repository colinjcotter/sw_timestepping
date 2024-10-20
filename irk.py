from sw_setup import *
from irksome import Dt, MeshConstant, RadauIIA, TimeStepper
MC = MeshConstant(mesh)

dT = MC.Constant(dt)
tc = MC.Constant(0.)

butcher_tableau = RadauIIA(2)

u0, h0 = fd.split(Un)
eqn = (
    fd.inner(v, Dt(u0))*dx
    + u_op(v, u0, h0)
    + phi*(Dt(h0))*dx
    + h_op(phi, u0, h0)
)

pc = {
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_pc_patch_construct_dim": 0,
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_local_type": "additive",
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_precompute_element_tensors": True,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "ilu",
    "patch_sub_pc_factor_mat_ordering_type": "rcm",
}

nomgparameters = {
    "snes_monitor": None,
    "snes_lag_preconditioner": 10,
    "snes_lag_preconditioner_persists": None,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    #"ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    #"ksp_view": None,
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

stepper = TimeStepper(eqn, butcher_tableau, tc, dT, Un,
                      solver_parameters=nomgparameters)

tdump = 0.

from firedrake.output import VTKFile
file_sw = VTKFile(name+'.pvd')
u0, h0 = Un.subfunctions
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)

nsteps = tcheck(tmax, dt)
step = 0
t = 0.

while t < tmax - 0.5*dt:
    step += 1

    PETSc.Sys.Print(f"\nTimestep {step} at time {t}, {t/tmax} of total\n")

    tdump += dt
    t += dt
    stepper.advance()

    if args.one_step:
        step = nsteps-1

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt

PETSc.Sys.Print("dt", dt, "ref_level", args.ref_level, "tmax", args.tmax)
assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"

etan.assign(h0 - H + b)
un.assign(u0)
checkpoint_output(un, etan)
