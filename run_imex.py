levels = ["6"]
imex_dts = [1] #18.75, 37.5, 75, 100]
dts = imex_dts
tmax = 86400
williamson=6
ncpus = 16

rows = []
import os

for dt in dts:
    for level in levels:
        options = {"tmax": tmax,
                   "williamson": 6,
                   "dt": dt,
                   "ref_level": level}

        args = []
        for key, value in options.items():
            args += ["--"+str(key), str(value)]
            fname = "imex_data_"+hex(abs(hash(str(options))))
            try:
                os.remove(fname)
            except:
                pass

        options["directory"] = fname
            
        os.makedirs(fname)
        args += ["--show_args"]
        args += ["--checkpointfile", fname+"/chk.h5"]
        args += ["--filename", fname+"/data"]
        args += ["-log_view", ":"+fname+"/log"]
        args += ["--coords_degree", "3"]
        args += ["&>", fname+"/out"]
        print("mpiexec -n "+str(ncpus)+" python imex.py " + " ".join(args))
        print("grep Main "+fname+"/log &> "+fname+"/stats")
        print("cat "+fname+"/chk.h5.out >> "+fname+"/stats")

        rows.append(options)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("imex.csv")
