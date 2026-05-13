import pandas as pd
from get_error import get_error

df = pd.read_csv("imex.csv")
times = []
setup = []
eta_errors = []
u_errors = []
for directory in df["directory"]:
    with open(directory+"/stats") as f:
        for n, line in enumerate(f):
            if n == 0: # get the overall timings
                print(line.split())
                times.append(float(line.split()[2]))
            if n == 2: # get the setup timings
                print(line.split())
                setup.append(float(line.split()[3]))
        file0 = directory+"/chk.h5"
        ref = "irk_ref_8May"
        eta_error, u_error = get_error(file0,
                                       ref+"/chk.h5")
        eta_errors.append(eta_error)
        u_errors.append(u_error)

df["Times"] = pd.Series(times)
df["Setup Time"] = pd.Series(setup)
df["Elevation error"] = eta_errors
df["Velocity error"] = u_errors

df.to_csv('imex_w_stats.csv')
