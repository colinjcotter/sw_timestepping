from irksome import RadauIIA, TimeStepper, GaussLegendre, \
    IRKAuxiliaryOperatorPC

from sw_setup import *
import numpy as np

if args.rk_type == 'RadauIIA':
    butcher_tableau = RadauIIA(args.rk_stages)
elif args.rk_type == 'GaussLegendre':
    butcher_tableau = GaussLegendre(args.rk_stages)

u0, h0 = fd.split(Un)
eqn = (
    fd.inner(v, Dt(u0))*dx
    + u_op(v, u0, h0)
    + phi*(Dt(h0))*dx
    + h_op(phi, u0, h0)
    + gamma*(fd.div(v)*Dt(h0)*dx
             + h_op(fd.div(v), u0, h0)
             )
)

monoparameters = {
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
    "ksp_ksp_richardson_scale": 0.8,
    "ksp_ksp_max_it": 2,
    "ksp" : patch
}

from irksome.pc import ldu
L, D, U = ldu(butcher_tableau.A)
Atilde_diag = np.diag(L@D)

class wavePC(fd.AuxiliaryOperatorPC):
    def form(self, pc, trial, test):
        u, h = fd.split(trial)
        v, q = fd.split(test)
        prefix = pc.getOptionsPrefix()
        stage_prefix = prefix + f"pc_stage"
        stage = PETSc.Options().getInt(stage_prefix)
        c = fd.Constant(Atilde_diag[stage]*dt)
        op = (
            fd.inner(v, u)*dx
            + c*(fd.inner(v, f*perp(u))*dx
                 - fd.div(v)*g*h*dx)
            + q*(h
                 + c*H*fd.div(u))*dx
             )
        return op, None

waveranaparameters = {
    #"snes_monitor": None,
    "snes_converged_reason": None,
    #"snes_lag_preconditioner": -2,
    #"snes_lag_preconditioner_persists": None,
    "snes_linesearch_type": "basic",
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "snes_rtol": args.ntol,
    "snes_ksp_ew": None,
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    #"ksp_converged_rate": None,
    "ksp_type": "fgmres",
    "ksp_rtol": args.ktol,
    "ksp_atol": 1e-50,
    "ksp_max_it": 60,
    #"ksp_view": None,
    "pc_type": "python",
    "pc_python_type": "irksome.RanaLD",
    "aux" : {
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
    }
}

for i in range(args.rk_stages):
    waveranaparameters[f"aux_pc_fieldsplit_{i}_fields"]=f"{2*i},{2*i+1}"
    waveranaparameters[f"aux_fieldsplit_{i}_ksp_type"]="gmres"
    waveranaparameters[f"aux_fieldsplit_{i}_ksp_monitor"]=None
    waveranaparameters[f"aux_fieldsplit_{i}_ksp_atol"]=0.
    waveranaparameters[f"aux_fieldsplit_{i}_ksp_rtol"]=1.0e-3
    waveranaparameters[f"aux_fieldsplit_{i}_pc_type"]="python"
    waveranaparameters[f"aux_fieldsplit_{i}_pc_python_type"]=f"{__name__}.wavePC"
    waveranaparameters[f"aux_fieldsplit_{i}_pc_stage"]=i
    waveranaparameters[f"aux_fieldsplit_{i}_aux_pc_type"]="lu"
    waveranaparameters[f"aux_fieldsplit_{i}_aux_pc_factor_mat_solver_type"]="mumps"

if args.pcscheme == "mono":
    parameters = monoparameters
elif args.pcscheme == "mg":
    parameters = mgparameters
elif args.pcscheme == "waverana":
    parameters = waveranaparameters
elif args.pcscheme == "rana":
    parameters = ranaparameters
elif args.pcscheme == "al":
    parameters = alparameters
else:
    raise NotImplementedError

class IRKMassPC(IRKAuxiliaryOperatorPC):
    def getNewForm(self, pc, u0, test):
        print(u0.function_space)
        print(test.function_space)
        _, p0 = fd.split(u0)
        return gamma*test*p0*dx

stepper = TimeStepper(eqn, butcher_tableau, tc, dT, Un,
                      solver_parameters=parameters)
#stepper.solver.set_transfer_manager(transfermanager)

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

for step in range(nsteps):
    PETSc.Sys.Print(f"\nTimestep {step} of {nsteps}.\n")

    tdump += dt
    t += dt
    stepper.advance()
    stepper.stages.assign(0.)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    itcount += stepper.solver.snes.getLinearSolveIterations()

    if args.one_step:
        t = tmax
        break
    
assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"

etan.assign(h0 - H + b)
un.assign(u0)
checkpoint_output(un, etan)
comm = PETSc.Sys.getDefaultComm()
if comm.rank == 0:
    with open(args.checkpointfile+'.out', 'w') as f:
        print("Iterations per step", itcount/nsteps, file=f) 
