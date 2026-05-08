import pandas as pd
from get_error import get_error

df = pd.read_csv("imex_convergent.csv")
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
        file0 = directory+"/chk.h5"
        ref = "irk_VI_ref"
        eta_error, u_error = get_error(file0,
                                       ref+"/chk.h5")
        eta_errors.append(eta_error)
        u_errors.append(u_error)

df["Times"] = pd.Series(times)
df["Elevation error"] = eta_errors
df["Velocity error"] = u_errors

df.to_csv('imex_w_stats.csv')
