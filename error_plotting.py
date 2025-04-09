from firedrake import *
from firedrake.__future__ import interpolate
import matplotlib.pyplot as plt


# folder = 'data1/'
folder = 'data1/w6/'
folder = 'data1/w6/Level_6/'
tmax = 86400

method = 'imex_w6'
# method = 'irk'
# method = 'irk_RadauIIA3'
# method = 'irk_GaussLegendre2'
# method = irk_w6_GaussLegendre3_L5_dt_150_tmax_86400
# method = 'irk_w6_GaussLegendre3'
# method = 'irk_w6_GaussLegendre1'
# method = 'irk_w6_GaussLegendre2'


# ref_method = 'irk_w6_GaussLegendre3'
ref_method = 'irk_w6_GaussLegendre1'
ref_time = 1

resol_spaces = {'L5': 'L5', 'L6': 'L6'}

if method == 'imex':
    resol_time = {'L5': [450, 225, 112.5, 56.25, 22.5, 1],
                  # 'L6': [225, 112.5, 56.25, 22.5, 11.25, 1]
                  'L6': [100, 50, 25, 12.75, 5.625]
                  }

elif method == 'imex_w6':
    resol_time = {#'L5': [150, 75, 37.5, 18.75],
                  'L5': [150, 75, 37.5, 18.75, 9.375],
                  # 'L6': [225, 112.5, 56.25, 22.5, 11.25, 1]
                  # 'L6': [100, 50, 25, 11.25, 5.625]}
                  'L6': [100, 75, 37.5, 18.75]}  # values not available yet!k

elif method == 'irk' or 'irk_RadauIIA3':
    resol_time = {'L5': [3600, 1800, 900, 450],
                  'L6': [3600, 1800, 900, 450]}

elif method == 'irk_w6_GaussLegendre1' or 'irk_w6_GaussLegendre2' or 'irk_w6_GaussLegendre3':
    resol_time = {'L5': [1200, 600, 300, 150, 75],
                  # 'L5': [1200, 600, 300, 150],
                  'L6': [1200, 600, 300, 150, 75]
                  }




# space_res = 'L5'
space_res = 'L6'

def filename(i,space_res):
    return folder + method + '_' + resol_spaces[space_res] + f'_dt_{resol_time[resol_spaces[space_res]][i]}_tmax_{tmax}.h5'

def filename_ref(space_res):
    return folder + ref_method + '_' + resol_spaces[space_res] + f'_dt_{ref_time}_tmax_{tmax}.h5'

reference = filename_ref(space_res)


errors_eta = []
errors_vel = []

#store values in dictionary
errors_eta_dict = {}

print('Ref file name:', reference)

for i in range(len(resol_time[resol_spaces[space_res]])):
    print(filename(i, space_res))
    #reference = filename(-1, space_res)

    with CheckpointFile(reference, 'r') as afile:
        mesh1 = afile.load_mesh("errormesh")
        eta1 = afile.load_function(mesh1, "Elevation")
        u1 = afile.load_function(mesh1, "Velocity")

    with CheckpointFile(filename(i, space_res), 'r') as afile:
        mesh = afile.load_mesh("errormesh")
        eta2 = afile.load_function(mesh, "Elevation")
        u2 = afile.load_function(mesh, "Velocity")

    V2 = eta2.function_space()
    V1 = u2.function_space()
    eta1_2 = assemble(interpolate(eta1, V2))
    u1_2 = assemble(interpolate(u1, V1))

    error_eta = errornorm(eta2, eta1_2) / norm(eta1)
    errors_eta.append(error_eta)
    error_vel = errornorm(u2, u1_2) / norm(u1)
    errors_vel.append(error_vel)


    print(i, '\n Full list of errors:')
    print(errors_eta)
    print(errors_vel)


# plot those values:
t = np.array(resol_time[space_res][:])
plt.loglog(resol_time[space_res][:],  errors_eta[:], 'k.-', label = space_res + '_eta')
plt.loglog(resol_time[space_res][:],  errors_vel[:], 'r.-', label = space_res + '_vel')
plt.loglog(t, 8e-12*t**2, '--', label = '2nd order' )
plt.loglog(t, 5e-15*t**3, '--', label = '3rd order' )
plt.loglog(t, 4e-18*t**4, '--', label = '4th order' )
plt.loglog(t, 3e-21*t**5, '--', label = '5th order' )
plt.loglog(t, 1e-24*t**6, '--', label = '6th order' )
plt.title(method)
plt.xlabel('dt')
plt.ylabel('L2 error')
plt.legend()
plt.show()


##

import sys
sys.exit()


# saved values from 15/11/2024

####################    IMEX     ####################################
## IMEX L6:
# reference is L6_1s
errors_eta_L6 = [np.float64(0.0019098588021198674),
 np.float64(0.0008696506294946101),
 np.float64(0.000345173459920019),
 np.float64(6.62814825890496e-05),
 np.float64(1.6608952761432457e-05),
 np.float64(1.857106149686142e-16)]


errors_vel_L6 = [np.float64(0.004938612786680899),
 np.float64(0.001792220667285171),
 np.float64(0.0006940854800443425),
 np.float64(0.0001974369800665682),
 np.float64(3.1971976694006375e-05),
 np.float64(1.829134504522177e-16)]

errors_eta_L5 =[np.float64(0.004938612786680899),
 np.float64(0.001792220667285171),
 np.float64(0.0006940854800443425),
 np.float64(0.0001974369800665682),
 np.float64(3.1971976694006375e-05),
 np.float64(1.829134504522177e-16)]

errors_vel_L5 = [np.float64(0.004980866476301717),
 np.float64(0.0018434274164471382),
 np.float64(0.0007141635385614361),
 np.float64(0.00020362521028415072),
 np.float64(3.3017184255040003e-05),
 np.float64(5.812756245472807e-16)]


#######################     IRK    ###################################
#irk:

errors_eta_L5 = [np.float64(0.008536036545316457),
 np.float64(0.005663071613930224),
 np.float64(0.0021506623280768255),
 np.float64(0.0006864038036116199)]

errors_vel_L5 = [np.float64(0.010345066169804634),
 np.float64(0.0065756625241595165),
 np.float64(0.0022841928251069042),
 np.float64(0.0007171486833525681)]



# ref 11.25s, NOT 1s yet because not ready yet!
errors_eta_L6 = [np.float64(0.008691923030772998),
 np.float64(0.005768375747764559),
 np.float64(0.0022241769959930394),
 np.float64(0.0007871259503174455)]

errors_vel_L6 = [np.float64(0.010334029801460608),
 np.float64(0.00658321071755304),
 np.float64(0.002325255489653287),
 np.float64(0.0008168956786154959)]









# plot those values:
t = np.array(resol_time[space_res][:-1])
plt.loglog(resol_time['L5'][:-1],  errors_eta_L5[:-1], 'k.-', label = 'L5 eta')
plt.loglog(resol_time['L5'][:-1],  errors_vel_L5[:-1], 'r.-', label = 'L5 vel')
plt.loglog(resol_time['L6'][:-1],  errors_eta_L6[:-1], 'b.-', label = 'L6 eta')
plt.loglog(resol_time['L6'][:-1],  errors_vel_L6[:-1], 'c.-', label = 'L6 vel')
plt.loglog(t, 1e-8*t**2, label = '2nd order' )
plt.xlabel('dt')
plt.ylabel('L2 error')
plt.legend()
plt.show()





















