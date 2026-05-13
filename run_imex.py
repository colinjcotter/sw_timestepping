levels = ["6"]
dts = [120, 100, 75, 37.5, 18.75]
day = 60*60*24
tmax = day*1
williamson=6
ncpus = [16]
vector_invariant = True

warmup = False

import subprocess, os

rows = []

for dt in dts:
    for level in levels:
        for ncpu in ncpus:
            options = {"tmax": tmax,
                       "williamson": 6,
                       "dt": dt,
                       "coords_degree": 1,
                       "ref_level": level,
                       }
            if vector_invariant:
                options["vector_invariant"] = ""

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
            print("grep 'Stepper:' "+fname+"/log > "+fname+"/stats")
            print("grep 'PCSetUp ' "+fname+"/log >> "+fname+"/stats")

            if vector_invariant:
                options["vector_invariant"] = "Y"
            else:
                options["vector_invariant"] = "N"
            rows.append(options)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("imex.csv")
