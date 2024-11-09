from sw_setup import *
from irksome import Dt, MeshConstant, RadauIIA, TimeStepper
from irksome.pc import RanaBase
import numpy as np

MC = MeshConstant(mesh)

dT = MC.Constant(dt)
tc = MC.Constant(0.)

butcher_tableau = RadauIIA(args.rk_stages)
class PQPC(RanaBase):
    def getAtilde(self, A):
        return np.diag(butcher_tableau.c)

u0, h0 = fd.split(Un)
eqn = (
    fd.inner(v, Dt(u0))*dx
    + u_op(v, u0, h0)
    + phi*(Dt(h0))*dx
    + h_op(phi, u0, h0)
)

if args.sdc:
    parameters = {"mat_type": "matfree",
                  "ksp_type": "gmres",
                  "ksp_monitor": None,
                  "ksp_atol": 1.0e-50,
                  "ksp_rtol": 1.0e-8,
                  "pc_type": "python",
                  "pc_python_type": "__main__.PQPC",
                  "aux" : 
                  {"pc_type": "fieldsplit",   # block preconditioner
                   "pc_fieldsplit_type": "additive"  # block diagonal
                   },
                  "snes_monitor": None,
                  "snes_lag_preconditioner": 5,
                  }

    per_field={
        "ksp_type": "preonly",
        "pc_type": "mg",
        "pc_mg_cycle_type": "v",
        "pc_mg_type": "multiplicative",
        "mg_levels_ksp_type": "gmres",
        "mg_levels_ksp_max_it": 3,
        #"mg_levels_ksp_convergence_test": "skip",
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.PatchPC",
        "mg_levels_patch_pc_patch_save_operators": True,
        "mg_levels_patch_pc_patch_partition_of_unity": True,
        "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
        "mg_levels_patch_pc_patch_construct_dim": 0,
        "mg_levels_patch_pc_patch_construct_type": "star",
        "mg_levels_patch_pc_patch_local_type": "additive",
        "mg_levels_patch_pc_patch_precompute_element_tensors": True,
        "mg_levels_patch_pc_patch_symmetrise_sweep": False,
        "mg_levels_patch_sub_ksp_type": "preonly",
        "mg_levels_patch_sub_pc_type": "lu",
        "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
    }
    
    for s in range(args.rk_stages):
        parameters["aux_pc_fieldsplit_"+str(s)+"_fields"] = \
            str(2*s)+","+str(2*s+1)
        parameters["aux_fieldsplit_%s" % (s,)] = per_field
else:
    pc = {
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
        "patch_pc_sub_ksp_type": "preonly",
    }
    parameters = {
        "mat_type": "matfree",
        "snes_monitor": None,
        "snes_stol": 1.0e-50,
        "snes_rtol": 1.0e-7,
        "snes_atol": 1.0e-50,
        "snes_converged_reason": None,
        #"snes_lag_preconditioner": 10,
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "ksp_atol": 1e-50,
        "ksp_rtol": 1e-8,
        "ksp_max_it": 400,
        "pc_type": "ksp",
        "ksp_ksp_type": "richardson",
        "ksp_ksp_richardson_scale": 1.,
        "ksp_ksp_rtol": 1e-10,
        "ksp_ksp_max_it": 1,
        "ksp_ksp_convergence_test": 'skip',
        "ksp_ksp_converged_maxits": None,
        "ksp" : pc
    }


stepper = TimeStepper(eqn, butcher_tableau, tc, dT, Un,
                      solver_parameters=parameters)
stepper.solver.set_transfer_manager(transfermanager)

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

for step in range(nsteps):
    PETSc.Sys.Print(f"\nTimestep {step} of {nsteps}.\n")

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
