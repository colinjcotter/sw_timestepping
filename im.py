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

uh = (u0+u1)*half
hh = (h0+h1)*half
    
eqn = (
    fd.inner(v, u1 - u0)*dx
    + dT*u_op(v, uh, hh, system="full")
    + phi*(h1 - h0)*dx
    + dT*h_op(phi, uh, hh, system="full")
)


class wavePC(fd.AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        u, p = fd.split(trial)
        v, q = fd.split(test)
        inner = fd.inner; div = fd.div
        a = (
            fd.inner(u, v)*fd.dx +
            dT*u_op(v, u, p, system="linear")
            + p*q*fd.dx +
            dT*h_op(q, u, p, system="linear")
        )
        #Returning None as bcs
        return (a, None)

parameters = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "snes_rtol": args.ntol,
    # "snes_max_it": 1,
    # "snes_convergence_test": "skip",
    "snes_lag_jacobian": -2,
    #"snes_lag_jacobian_persists": None,
    "ksp_monitor": None,
    "ksp_converged_rate": None,
    # "ksp_view": None,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_atol": 1e-50,
    "ksp_max_it": 60,
    "pc_type": "ksp",
    "ksp_ksp_type": "richardson",
    "ksp_ksp_richardson_scale": 0.95,
    "ksp_ksp_max_it": 2,
    "ksp_pc_type": "python",
    "ksp_pc_python_type": "firedrake.PatchPC",
    "ksp_patch_pc_patch_save_operators": True,
    "ksp_patch_pc_patch_partition_of_unity": True,
    "ksp_patch_pc_patch_sub_mat_type": "seqdense",
    "ksp_patch_pc_patch_construct_dim": 0,
    "ksp_patch_pc_patch_construct_type": "star",
    "ksp_patch_pc_patch_local_type": "additive",
    "ksp_patch_pc_patch_precompute_element_tensors": True,
    "ksp_patch_pc_patch_symmetrise_sweep": False,
    "ksp_patch_sub_ksp_type": "preonly",
    "ksp_patch_sub_pc_type": "lu",
    "ksp_patch_sub_pc_factor_shift_type": "nonzero"
}


semiparameters = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "snes_rtol": args.ntol,
    # "snes_max_it": 1,
    # "snes_convergence_test": "skip",
    "snes_lag_jacobian": -2,
    #"snes_lag_jacobian_persists": None,
    "ksp_monitor": None,
    "ksp_converged_rate": None,
    # "ksp_view": None,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_atol": 1e-50,
    "ksp_max_it": 60,
    "pc_type": "composite",
    "pc_composite_type": "multiplicative",
    "pc_composite_pcs": "python,python",
    "sub_0": {"pc_python_type": "__main__.wavePC",
              "aux_pc_type": "python",
              "aux_pc_python_type": "firedrake.HybridizationPC",
              "aux_hybridization": {
                  "ksp_type": "preonly",
                  "pc_type": "lu"
                  }
              },
    "sub_1" : {
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
}


nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob, options_prefix="swe",
                                        solver_parameters=semiparameters)
nsolver.set_transfer_manager(transfermanager)

Unp1.assign(Un)

tdump = 0.
t = 0.
PETSc.Sys.Print('tmax', tmax, 'dt', dt)

if args.hybrid:
    u0, h0, ll0 = Un.subfunctions
else:
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

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    itcount += nsolver.snes.getLinearSolveIterations()
    if args.one_step:
        break

PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/nsteps,
                "dt", dt, "ref_level", args.ref_level, "tmax", args.tmax)
if not args.one_step:
    assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"

etan.assign(h0 - H + b)
un.assign(u0)
checkpoint_output(un, etan)
