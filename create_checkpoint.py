# script to create a checkpoint file for sw timestepping experiments
from sw_setup import *

if args.checkpointfile == 'none':
    raise ValueError('Need to specify a checkpoint file.')

with fd.CheckpointFile(args.checkpointfile, 'w') as afile:
    afile.save_mesh(mesh)
