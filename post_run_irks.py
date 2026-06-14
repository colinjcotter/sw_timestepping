import pandas as pd
from get_error import get_error

df = pd.read_csv("irks.csv")
ref = "/home/cjc1/repositories/sw_timestepping/irk_ref_8May"
its = []
first = []
rest = []
eta_errors = []
u_errors = []
for i, directory in enumerate(df["directory"]):
    with open(directory+"/stats") as f:
        for n, line in enumerate(f):
            if n == 0: # get the timings for the first step
                first.append(float(line.split()[2]))
            if n == 1: # get the timings for the rest of the steps
                rest.append(float(line.split()[2]))
                rest[-1] += first[-1]
            if n == 2: # get the iteration counts
                its.append(float(line.split()[5]))
        while len(first) < i+1:
            first.append(-666)
        while len(rest) < i+1:
            rest.append(-666)
        while len(its) < i+1:
            its.append(-666)
        file0 = directory+"/chk.h5"
        try:
            eta_error, u_error = get_error(file0,
                                           ref+"/chk.h5")
        except:
            eta_error = -666
            u_error = -666

        eta_errors.append(eta_error)
        u_errors.append(u_error)
        print(i, len(its), len(first), len(rest), 'lens')
print(its)
df["First Step Time"] = pd.Series(first)
df["Time"] = pd.Series(rest)
df["Iterations per timestep"] = pd.Series(its)
df["Elevation error"] = eta_errors
df["Velocity error"] = u_errors
df.to_csv('irks_w_stats.csv')
