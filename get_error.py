from firedrake import *

def get_error(error_chkpoint, chkpoint):
    #read in mesh and data from chkpoint

    print(chkpoint)
    with CheckpointFile(chkpoint, 'r') as afile:
        meshr = afile.load_mesh("errormesh")
        etar = afile.load_function(meshr, "Elevation")
        ur = afile.load_function(meshr, "Velocity")

    print(error_chkpoint)
    with CheckpointFile(error_chkpoint, 'r') as afile:
        meshi = afile.load_mesh("errormesh")
        etai = afile.load_function(meshi, "Elevation")
        ui = afile.load_function(meshi, "Velocity")

    etai_r = Function(etar.function_space()).interpolate(etai)
    ui_r = Function(ur.function_space()).interpolate(ui)

    eta_error = norm(etai_r-etar)/norm(etar)
    u_error = norm(ui_r-ur)/norm(ur)

    return eta_error, u_error
