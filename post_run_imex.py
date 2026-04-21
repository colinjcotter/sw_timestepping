import pandas as pd

df = pd.read_csv("imex.csv")
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
    df["Times"] = pd.Series(times)

df.to_csv('imex_w_stats.csv')
