levels = ["6"]
#dts = [14400, 10800, 7200, 3600, 2400, 1200, 600, 300]
dts = [3600]
pcs = ["mg"]
#stages = [1,2,3] # overridden in the presence of DIRKs, best not to mix DIRK and FIRK
stages = [1]
day = 60*60*24
tmax = day*1
ntol = 1.0e-6
kspatol = 2.0e2
williamson=6
ncpus = [16]
vector_invariant = True

#irks = [("WSODIRK", 4, 3, 2),
#        ("WSODIRK", 4, 3, 3),
#        "Alexander"]
irks = ["GaussLegendre"]

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
                                   "ntol": ntol,
                                   "kspatol": kspatol,
                                   "williamson": 6,
                                   "dt": dt,
                                   "coords_degree": 1,
                                   "ref_level": level,
                                   "rk_stages": stage,
                                   "pcscheme": pc,
                                   "rk_type": irk_name,
                                   }
                        if vector_invariant:
                            options["vector_invariant"] = ""
                        if irk_name == "WSODIRK":
                            options["rk_stages"] = irk[1]
                            options["WSODIRK_order"] = irk[2]
                            options["weak_stage_order"] = irk[3]
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
                        print("grep 'Main Stage:' "+fname+"/log > "+fname+"/stats")
                        print("grep Iterations "+fname+"/out >> "+fname+"/stats")
                        if vector_invariant:
                            options["vector_invariant"] = "Y"
                        else:
                            options["vector_invariant"] = "N"
                        rows.append(options)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("irks.csv")
