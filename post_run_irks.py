import pandas as pd

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
                its.append(vals[0])
    df["Times"] = pd.Series(times)
    df["Iterations per timestep"] = pd.Series(its)

df.to_csv('irks_w_stats.csv')
