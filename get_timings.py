level = "6"
imex_dts = [18.75, 37.5, 75, 100]
irk_dts = [37.5, 75, 150, 300, 600, 1200, 2400]
dts = irk_dts
script = "irk"
#script = "imex"
irk = "GaussLegendre"
#irk = "RadauIIA"
stages = 3
tmax = 86400
ntol =1.0e-6
williamson=6
ncpus = [16]

for ncpu in ncpus:
    for dt in dts:
        dt = str(dt)
        if script == "irk":
            name = script+"_w"+str(williamson)+"_"+irk+str(stages)+"_L"+level+"_dt_"+str(dt)+"_tmax_"+str(tmax)
        else:
            name = script+"_w"+str(williamson)+"_L"+level+"_dt_"+str(dt)+"_tmax_"+str(tmax)
        name += "_ntol_1e-06_cpu_"+str(ncpu)+".out"
        print("echo "+name)
        print("grep -i main "+ name)
