import pandas as pd
from get_error import get_error

df = pd.read_csv("imex.csv")
firsttimes = []
tottimes = []
eta_errors = []
u_errors = []
for directory in df["directory"]:
    with open(directory+"/stats") as f:
        for n, line in enumerate(f):
            if n == 0: # get the first timestep timings
                try:
                    print(line.split())
                    firsttimes.append(float(line.split()[2]))
                except:
                    firsttimes.append(-666)
            if n == 1: # get the second timestep timings
                try:
                    print(line.split())
                    tottimes.append(float(line.split()[2]))
                    tottimes[-1] += firsttimes[-1]
                except:
                    setup.append(-666)
        file0 = directory+"/chk.h5"
        ref = "irk_ref_8May"
        try:
            eta_error, u_error = get_error(file0,
                                           ref+"/chk.h5")
            eta_errors.append(eta_error)
            u_errors.append(u_error)
        except:
            eta_errors.append(-666)
            u_errors.append(-666)

df["Time"] = pd.Series(tottimes)
df["First Step Time"] = pd.Series(firsttimes)
df["Elevation error"] = eta_errors
df["Velocity error"] = u_errors

df.to_csv('imex_w_stats.csv')
