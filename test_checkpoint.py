from firedrake import *
from firedrake.__future__ import interpolate

with CheckpointFile("data1/imex_L6_dt_22.5.h5", 'r') as afile:
    mesh1 = afile.load_mesh("errormesh")
    eta1 = afile.load_function(mesh1, "Elevation")
    u1 = afile.load_function(mesh1, "Velocity")


with CheckpointFile("data1/imex_L6_dt_56.25.h5", 'r') as afile:
    mesh = afile.load_mesh("errormesh")
    eta2 = afile.load_function(mesh, "Elevation")
    u2 = afile.load_function(mesh, "Velocity")

V2 = eta2.function_space()
V1 = u2.function_space()

eta1_2 = assemble(interpolate(eta1, V2))
u1_2 = assemble(interpolate(u1, V1))

print(errornorm(eta2, eta1_2)/norm(eta2))
print(errornorm(u2, u1_2)/norm(u2))
