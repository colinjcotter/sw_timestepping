# Testing the imex stepper using advection-diffusion
from imex_stepper import ARK3222
import firedrake as fd
from petsc4py import PETSc

n = 100
mesh = fd.PeriodicUnitIntervalMesh(n)
V = fd.FunctionSpace(mesh, "CG", 2)
Un = fd.Function(V)
dx = fd.dx

def nonlinear(V, U):
    return V*U.dx(0)*dx
    
def linear(V, U):
    return V.dx(0)*U.dx(0)*dx
    
lparams = {}
massparams = {}

dt = 0.001
dT = fd.Constant(dt)

stepper = ARK3222(linear, nonlinear, Un, dT, lparams, massparams)

x, = fd.SpatialCoordinate(mesh)
u0 = fd.sin(fd.pi*2*x)
# analytic solution
# U = Im(u(t)*exp(2*pi*i*x)), u(0) = 1
# u_t = (-2*pi*i - 4*pi**2)*u
# has solution u = exp((-2*pi*i - 4*pi**2)t)
# U = Im(exp(-(2*pi*i + 4*pi**2)t + 2*pi*i*x))
#   = exp(-4*pi**2*t)*Im(exp(-2*pi*i*(x-t)))
#   = exp(-4*pi**2*t)*sin(2*pi*(x-t))
tmax = 0.01
exact = fd.exp(-4*fd.pi**2*tmax)*(fd.sin(2*fd.pi*(x-tmax)))

dts = [tmax*2**(-k) for k in range(5)]
errors = []

for dt in dts:
    t = 0.
    dT.assign(dt)
    Un.interpolate(u0)    
    while t < tmax - 0.5*dt:
        PETSc.Sys.Print(f"\nTime {t}, {t/tmax} of total\n")
        t += dt

        stepper.advance()

    assert abs(t-tmax) < 1.0e-5, "t is not equal to tmax"
    error = float(fd.norm(Un - exact))
    errors.append(error)

import numpy as np
dts = np.array(dts)
errors = np.array(errors)
print(dts)
print(errors)
order = -np.log(errors[1:]/errors[0:-1])/np.log(2)
print(order)
