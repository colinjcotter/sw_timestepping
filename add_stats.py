import pandas as pd

filename = "irks2.csv"
irks = True

df = pd.read_csv(filename)

df["time"] = None
if irks:
    df["its per step"] = None

for idx, value in df["directory"].items():
    print(value)
    with open(value+"/stats") as f:
        timeline = f.readline().split()
        try:
            assert timeline[2] == "Stage:"
        except:
            time0 = -666
        else:
            time0 = float(timeline[3])
        df.at[idx, "time"] = time0
        if irks:
            f.readline().split()
            f.readline().split()
            its = f.readline().split()
            try:
                assert(its[2] == 'step')
            except:
                its = -666
            else:
                its = float(its[3])
            df.at[idx, "its per step"] = its

df.to_csv(filename)
