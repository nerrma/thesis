#!/usr/bin/env python3

# import og_ista as og
import og_gd as og
import jax_gd as jx
import pytorch_gd

# import jax_ista as jx
import numpy as np
import time

import jax

jax.config.update("jax_platform_name", "cpu")

n = 250
p = 20
n_iter = 2

# print("OG GD: ")
# start = time.time()
# err = og.run_sim(n, p, n_iter=n_iter)
# end = time.time()
# print(f"\tIACV {np.mean(err['IACV'])}")
# print(f"\tNS {np.mean(err['NS'])}")
# print(f"\tIJ {np.mean(err['NS'])}")
# print(f"\that {np.mean(err['hat'])}")
# print(f"time taken : {end - start}")

print("JAX GD: ")
start = time.time()
err, cv_err, runtime = jx.run_sim(n, p, n_iter=n_iter)
end = time.time()
print(f"\tIACV {np.mean(err['IACV'])}")
print(f"\tNS {np.mean(err['NS'])}")
print(f"\tIJ {np.mean(err['NS'])}")
print(f"\that {np.mean(err['hat'])}")
print(f"time taken : {end - start}")

# print("PYTORCH GD: ")
# start = time.time()
# err = pytorch_gd.run_sim(n, p, n_iter=n_iter)
# end = time.time()
# print(f"\tIACV {np.mean(err['IACV'])}")
# print(f"\tNS {np.mean(err['NS'])}")
# print(f"\tIJ {np.mean(err['NS'])}")
# print(f"\that {np.mean(err['hat'])}")
# print(f"time taken : {end - start}")
