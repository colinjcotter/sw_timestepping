import pandas as pd
from get_error import get_error

df = pd.read_csv("irks.csv")
ref = "imex_data_0x43652b16413a9c67"
its = []
times = []
eta_errors = []
u_errors = []
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
        eta_error, u_error = get_error(directory+"/chk.h5",
                                       ref+"/chk.h5")
        eta_errors.append(eta_error)
        u_errors.append(u_error)
df["Times"] = pd.Series(times)
df["Iterations per timestep"] = pd.Series(its)
df["Elevation error"] = eta_errors
df["Velocity error"] = u_errors
df.to_csv('irks_w_stats.csv')
