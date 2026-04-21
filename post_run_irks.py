import pandas as pd
from get_error import get_error

df = pd.read_csv("irks.csv")
its = []
times = []
for directory in df["directory"]:
    with open(directory+"/stats") as f:
        for n, line in enumerate(f):
            if n == 0: # get the timings
                vals = []
                for s in line.split():
                    try:
                        vals.append(float(s))
                    except:
                        continue
                times.append(vals[0])
            if n == 1: # get the KSP it count
                vals = []
                for s in line.split():
                    try:
                        vals.append(float(s))
                    except:
                        continue
                its.append(vals[1])
    df["Times"] = pd.Series(times)
    df["Iterations per timestep"] = pd.Series(its)
    print(directory)
    eta_error, u_error = get_error(directory+"/chk.h5",
                                   "rk_w6_GaussLegendre1_L6_dt_1_tmax_86400.h5")
    df["Elevation error"] = eta_error
    df["Velocity error"] = u_error
df.to_csv('irks_w_stats.csv')
