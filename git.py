from irksome import Dt, MeshConstant, GalerkinTimeStepper
from irksome.pc import RanaBase

from sw_setup import *
import numpy as np

MC = MeshConstant(mesh)

dT = MC.Constant(dt)
tc = MC.Constant(0.)

u0, h0 = fd.split(Un)
eqn = (
    fd.inner(v, Dt(u0))*dx
    + u_op(v, u0, h0)
    + phi*(Dt(h0))*dx
    + h_op(phi, u0, h0)
)

starasm = {
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_sub_sub_pc_type": "lu",
    "assembled_pc_star_sub_sub_ksp_type": "preonly",
    "assembled_pc_star_construct_dim": 0,
    #"assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm"
    "assembled_pc_star_backend": "tinyasm",
}

patch = {
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_pc_patch_construct_dim": 0,
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_local_type": "additive",
    "patch_pc_patch_precompute_element_tensors": True,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
    #"patch_sub_pc_factor_shift_type": "nonzero"
}

parameters = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_linesearch_type": "basic",
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "snes_rtol": args.ntol,
    # "snes_max_it": 1,
    # "snes_convergence_test": "skip",
    "snes_lag_jacobian": 40,
    "snes_lag_jacobian_persists": None,
    "snes_ksp_ew": None,
    "ksp_monitor": None,
    "ksp_converged_rate": None,
    # "ksp_view": None,
    "ksp_type": "gcr",
    "ksp_rtol": args.ktol,
    "ksp_atol": 1e-50,
    "ksp_max_it": 60,
    "pc_type": "ksp",
    "ksp_ksp_type": "gmres",
    #"ksp_ksp_richardson_scale": 0.8,
    "ksp_ksp_max_it": 2,
    "ksp" : patch
}

stepper = GalerkinTimeStepper(eqn, args.rk_stages, tc, dT, Un,
                              basis_type="integral",
                              solver_parameters=parameters)
stepper.solver.set_transfer_manager(transfermanager)

tdump = 0.
tn = 0.

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

for step in range(nsteps):
    PETSc.Sys.Print(f"\nTimestep {step} of {nsteps}.\n")

    tdump += dt
    t += dt
    stepper.advance()
    stepper.stages.assign(0.)

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
