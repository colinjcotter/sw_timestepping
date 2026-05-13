from sw_setup import *
import numpy as np

dT = fd.Constant(dt)
tc = fd.Constant(0.)

stage_type = args.stage_type

if args.rk_type == 'RadauIIA':
    butcher_tableau = RadauIIA(args.rk_stages)
elif args.rk_type == 'GaussLegendre':
    butcher_tableau = GaussLegendre(args.rk_stages)
elif args.rk_type == 'WSODIRK':
    butcher_tableau = WSODIRK(args.rk_stages,
                              args.WSODIRK_order,
                              args.weak_stage_order)
    stage_type = "dirk"
elif args.rk_type == 'Alexander':
    butcher_tableau = Alexander()
    stage_type = "dirk"
else:
    raise NotImplementedError
    
u0, h0 = fd.split(Un)
eqn = (
    fd.inner(v, Dt(u0))*dx
    + u_op(v, u0, h0)
    + phi*(Dt(h0))*dx
    + h_op(phi, u0, h0)
)

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
    "patch_sub_pc_factor_shift_type": "nonzero"
}

starasm = {
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_sub_sub_pc_type": "lu",
    "assembled_pc_star_sub_sub_ksp_type": "preonly",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    "assembled_pc_star_backend": "tinyasm",
}

parameters = {
    "mat_type": "matfree",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_ksp_ew": None,
    "snes_atol": 0,
    "snes_stol": 0,
    "snes_rtol": args.ntol,
    "snes_lag_jacobian": 100,
    #"snes_lag_jacobian_persists": None,
    "ksp_converged_rate": None,
    "ksp_max_it": 60,
    #"ksp_view": None
}

if args.pcscheme == 'mg':
    mgparameters = {
        "ksp_type": "fgmres",
        "ksp_monitor": None,
        "pc_type": "mg",
        "pc_mg_cycle_type": "v",
        "pc_mg_type": "multiplicative",
        "mg_levels_ksp_type": "gmres",
        "mg_levels_ksp_max_it": 3,
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.PatchPC",
        "mg_levels_patch_pc_patch_save_operators": True,
        "mg_levels_patch_pc_patch_partition_of_unity": True,
        #"mg_levels_patch_pc_patch_sub_mat_type": "aij",
        "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
        "mg_levels_patch_pc_patch_construct_dim": 0,
        "mg_levels_patch_pc_patch_construct_type": "star",
        "mg_levels_patch_pc_patch_local_type": "additive",
        "mg_levels_patch_pc_patch_precompute_element_tensors": True,
        "mg_levels_patch_pc_patch_symmetrise_sweep": False,
        "mg_levels_patch_sub_ksp_type": "preonly",
        "mg_levels_patch_sub_pc_type": "lu",
        "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
        "mg_levels_patch_sub_pc_factor_mat_ordering_type": "rcm",
        "mg_levels_patch_sub_pc_factor_reuse_ordering" : None,
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
    }
    parameters = parameters | mgparameters
elif args.pcscheme == 'patch':
    pparameters = {
        "ksp_type": "gmres",
        "pc_type": "ksp",
        "ksp_ksp_type": "richardson",
        "ksp_ksp_richardson_scale": 0.5,
        "ksp_ksp_max_it": 2,
        "ksp" : starasm,
    }
    parameters = parameters | pparameters
else:
    raise NotImplementedError

stepper = TimeStepper(eqn, butcher_tableau, tc, dT, Un,
                      stage_type = stage_type,
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
itcount = 0

print = PETSc.Sys.Print

for step in range(nsteps):
    PETSc.Sys.Print(f"\nTimestep {step} of {nsteps}.\n")

    tdump += dt
    t += dt
    with PETSc.Log.Stage("Stepper"):
        stepper.stages.zero()
        F = stepper.solver._problem.F
        with fd.assemble(F).dat.vec_ro as vec:
            res0 = vec.norm()
        snes_rtol = stepper.solver.snes.rtol
        stepper.solver.snes.ksp.atol = 0.1*snes_rtol*res0
        stepper.advance()
    itcount += stepper.solver.snes.getLinearSolveIterations()

    if args.one_step:
        break

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt

if not args.one_step:
    assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"

PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/(step+1),
                "dt", dt, "ref_level", args.ref_level, "tmax", tmax)

etan.assign(h0 - H + b)
un.assign(u0)
checkpoint_output(un, etan)
