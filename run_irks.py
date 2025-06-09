levels = ["6"]
imex_dts = [18.75, 37.5, 75, 100]
irk_dts = [3600, 2400] #[37.5, 75, 150, 300, 600, 1200, 2400, 3600]
dts = irk_dts
script = "irk"
irks = ["GaussLegendre", "RadauIIA"]
stages = [1] #,2,3,4]
tmax = 86400
ntol = 1.0e-8
ktol = 1.0e-10
williamson=6
ncpus = [16]

import subprocess, os

data = []

for dt in dts:
    for level in levels:
        for stage in stages:
            args = ["--tmax", str(tmax), "--ntol", str(ntol),
                    "--ktol", str(ktol),
                    "--williamson", str(6), "--pcscheme", "mg"]
            args += ["--dt", str(dt)]
            args += ["--ref_level", str(level)]
            args += ["--rk_stages",str(stage)]

            fname = "irk_"+hex(abs(hash(str(args))))
            try:
                os.remove(fname)
            except:
                pass
            os.makedirs(fname)
            args += ["--checkpointfile", fname+"/chk.h5"]
            args += ["--filename", fname+"/data"]
            args += ["-log_view", ":"+fname+"/out"]
            args += ["-pcscheme", "mg"]
            subprocess.run(["mpiexec","-n","16","python","irk.py"]+args)

            with open(fname+"/chk.h5.out") as f:
                read_data = f.readline()
            read_data = read_data.rsplit(sep=" ")
            assert(read_data[2] == "step")
            args += ["Iterations per step", read_data[3].strip("\n")]

            with open(fname+"/out") as f:
                for line in f:
                    if "Main" in line:
                        ll = line.rsplit(sep=" ")
                        assert(ll[7] == "Main")
                        assert(ll[8] == "Stage:")
                        args += ["Time", ll[9]]
                        break
            data.append(args)

with open("experiments.dat", "w") as f:
    for line in data:
        f.write(str(line)+"\n")
