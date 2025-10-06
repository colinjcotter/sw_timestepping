level = "6"
imex_dts = [18.75, 37.5, 75, 100]
irk_dts = [37.5, 75, 150, 300, 600, 1200, 2400]
dts = irk_dts
#dts = imex_dts
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
            print("mpiexec -n "+str(ncpu)+" python3 "+script+".py --filename "+name+" --dt "+dt+" --ref_level "+level+" --checkpointfile "+name+".h5 --rk_type "+irk+" --rk_stages "+str(stages)+" --tmax "+str(tmax)+" --ntol "+str(ntol)+" --williamson "+str(williamson)+" -log_view &>"+name+"_cpu_"+str(ncpu)+".out")
        else:
            name = script+"_w"+str(williamson)+"_L"+level+"_dt_"+str(dt)+"_tmax_"+str(tmax)
            print("mpiexec -n "+str(ncpu)+" python3 "+script+".py --filename "+name+" --dt "+dt+" --ref_level "+level+" --checkpointfile "+name+".h5 --tmax "+str(tmax)+" --williamson "+str(williamson)+" -log_view &>"+name+"_cpu_"+str(ncpu)+".out")
