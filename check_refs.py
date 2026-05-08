import pandas as pd
from get_error import get_error

eta_error, u_error = get_error("imex_VI_ref/chk.h5",
                               "irk_VI_ref/chk.h5")

print(eta_error, u_error)
