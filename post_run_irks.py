import pandas as pd
from get_error import get_error

df = pd.read_csv("irks.csv")
ref = "irk_ref_8May"
its = []
times = []
setup = []
eta_errors = []
u_errors = []
for directory in df["directory"]:
    with open(directory+"/stats") as f:
        for n, line in enumerate(f):
            if n == 0: # get the overall timings
                times.append(float(line.split()[2]))
            if n == 2: # get the setup timings
                setup.append(float(line.split()[3]))
            if n == 3: # get the KSP it count
                vals = []
                for s in line.split():
                    try:
                        vals.append(float(s))
                    except:
                        continue
                its.append(vals[1])
        file0 = directory+"/chk.h5"
        try:
            eta_error, u_error = get_error(file0,
                                           ref+"/chk.h5")
        except:
            eta_error = -666
            u_error = -666
            
        eta_errors.append(eta_error)
        u_errors.append(u_error)
df["Times"] = pd.Series(times)
df["Setup Time"] = pd.Series(setup)
df["Iterations per timestep"] = pd.Series(its)
df["Elevation error"] = eta_errors
df["Velocity error"] = u_errors
df.to_csv('irks_w_stats.csv')
