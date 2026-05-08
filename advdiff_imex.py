# Testing the imex stepper using advection-diffusion
from imex_stepper import ARK3222
import firedrake as fd

n = 100
mesh = UnitPeriodicIntervalMesh(n)
V = fd.FunctionSpace(

Unp1 = fd.Function(W)
u1, h1 = fd.split(Unp1)

half = fd.Constant(0.5)
quarter = fd.Constant(0.25)

u0, h0 = fd.split(Un)

energy_expr = 0.5*(fd.inner(u0, u0)*h0 + g*(h0**2 + h0*b))*fd.dx

def nonlinear(V, U):
    v, phi = fd.split(V)
    u, h = fd.split(U)
    eqn = u_op(v, u, h, system="nonlinear")
    eqn += h_op(phi, u, h, system="nonlinear")
    return eqn
    
def linear(V, U):
    v, phi = fd.split(V)
    u, h = fd.split(U)
    eqn = u_op(v, u, h, system="linear")
    eqn += h_op(phi, u, h, system="linear")
    return eqn
    
lparams = {
    "mat_type": "matfree",
    "snes_lag_jacobian": -2,
    "snes_lag_jacobian_persists": None,
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_monitor": None,
    "pc_type": "python",
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {'ksp_type': 'preonly',
                      'pc_type': 'lu',
                      "pc_factor_mat_solver_type":'superlu_dist'
                      }}
    
mass = {
    "ksp_type": "gmres",
    "pc_type": "bjacobi",
    "sub_pc_type": "ilu"
}

massparams = {
    "snes_lag_jacobian": -2,
    "snes_lag_jacobian_persists": None,
    "snes_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "fieldsplit_0": mass,
    "fieldsplit_1": mass
}

stepper = ARK3222(linear, nonlinear, Un, dT, lparams, massparams)

tdump = 0.
t = 0.
PETSc.Sys.Print('tmax', tmax, 'dt', dt)

u0, h0 = Un.subfunctions

from firedrake.output import VTKFile
file_sw = VTKFile(name+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)

itcount = 0
energy0 = fd.assemble(energy_expr)
step = 0

while t < tmax - 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {step} at time {t}, {t/tmax} of total\n")
    step += 1
    t += dt
    tdump += dt

    stepper.advance()

    if args.one_step:
        step = nsteps-1

    energy = fd.assemble(energy_expr)
    PETSc.Sys.Print("relative energy error", (energy-energy0)/energy0)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
PETSc.Sys.Print("dt", dt, "ref_level", args.ref_level, "tmax", tmax)
assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"

etan.assign(h0 - H + b)
un.assign(u0)
checkpoint_output(un, etan)
