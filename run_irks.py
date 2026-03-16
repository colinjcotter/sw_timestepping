levels = ["5"]
dts = [14400, 10800, 7200, 3600, 2400, 1200, 600, 300]
pcs = ["mg"]
stages = [
tmax = 86400
ntol = 1.0e-6
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
                    options = {"tmax": tmax,
                               "ntol": ntol,
                               "williamson": 6,
                               "dt": dt,
                               "ref_level": level,
                               "rk_stages": stage,
                               "pcscheme": pc,
                               "rk_type": irk,
                               }
                    args = []
                    for key, value in options.items():
                        args += ["--"+str(key), str(value)]
                    fname = "irk_data_"+hex(abs(hash(str(options))))
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
                    print("mpiexec -n "+str(ncpu)+" python irk.py " + " ".join(args))
                    print("grep Main "+fname+"/log &> "+fname+"/stats")
                    print("cat "+fname+"/chk.h5.out >> "+fname+"/stats")

                    rows.append(options)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("irks.csv")
