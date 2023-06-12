#!/usr/bin/env python3

import og_gd
import pytorch_gd
import jax_gd
import numpy as np
import time

n = 100
p = 20
n_iter = 100

# print("OG GD: ")
# start = time.time()
# err = og_gd.run_sim(n, p, n_iter=n_iter)
# end = time.time()
# print(f"\tIACV {np.mean(err['IACV'])}")
# print(f"\tNS {np.mean(err['NS'])}")
# print(f"\tIJ {np.mean(err['NS'])}")
# print(f"\that {np.mean(err['hat'])}")
# print(f"time taken : {end - start}")

print("JAX GD: ")
start = time.time()
err = jax_gd.run_sim(n, p, n_iter=n_iter)
end = time.time()
print(f"\tIACV {np.mean(err['IACV'])}")
print(f"\tNS {np.mean(err['NS'])}")
print(f"\tIJ {np.mean(err['NS'])}")
print(f"\that {np.mean(err['hat'])}")
print(f"time taken : {end - start}")

print("PYTORCH GD: ")
start = time.time()
err = pytorch_gd.run_sim(n, p, n_iter=n_iter)
end = time.time()
print(f"\tIACV {np.mean(err['IACV'])}")
print(f"\tNS {np.mean(err['NS'])}")
print(f"\tIJ {np.mean(err['NS'])}")
print(f"\that {np.mean(err['hat'])}")
print(f"time taken : {end - start}")
