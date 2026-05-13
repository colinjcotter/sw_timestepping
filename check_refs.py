import pandas as pd
from get_error import get_error

eta_error, u_error = get_error("imex_ref_8May/chk.h5",
                               "irk_ref_8May/chk.h5")

print(eta_error, u_error)
