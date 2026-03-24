levels = ["6"]
dts = [14400, 10800, 7200, 3600, 2400, 1200, 600, 300]

day = 60*60*24
tmax = day*1
williamson=6
ncpus = [16]

warmup = False

import subprocess, os

rows = []

for dt in dts:
    for level in levels:
        for stage in stages:
            for ncpu in ncpus:
                for pc in pcs:
                    for irk in irks:
                        if isinstance(irk, tuple):
                            irk_name = irk[0]
                        else:
                            irk_name = irk
                        options = {"tmax": tmax,
                                   "williamson": 6,
                                   "dt": dt,
                                   "coords_degree": 2,
                                   "ref_level": level,
                                   }
                        args = []
                        for key, value in options.items():
                            args += ["--"+str(key), str(value)]
                        fname = "imex_data_"+hex(abs(hash(str(options))))
                        try:
                            os.remove(fname)
                        except:
                            pass

                        options["directory"] = fname
                        options["ncpus"] = ncpu
                        os.makedirs(fname)
                        if warmup:
                            args += ["--one_step"]
                        args += ["--show_args"]
                        args += ["--checkpointfile", fname+"/chk.h5"]
                        args += ["--filename", fname+"/data"]
                        args += ["-log_view", ":"+fname+"/log"]
                        args += ["&>", fname+"/out"]
                        print("mpiexec -n "+str(ncpu)+" python imex.py " + " ".join(args))
                        print("grep 'Main Stage:' "+fname+"/log > "+fname+"/stats")
                        print("grep Iterations "+fname+"/out >> "+fname+"/stats")

                        rows.append(options)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("imex.csv")
