#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import numpy as np

nx = 450
ny = 225
nz = 564
ppc_sum = 12 + 6
tnp_percent = 1.0

# nx.ny.nz.(sum of ppc for all species).tnp_percent / number_of_GPUs = 600,000,000
ngpu_total = np.ceil((nx*ny*nz*ppc_sum*tnp_percent)/600000000)

print("Total number of GPUs: ", int(ngpu_total))