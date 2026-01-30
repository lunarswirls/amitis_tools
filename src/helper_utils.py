#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd


def lon_diff(a, b):
    """
    Minimal angular difference in degrees.
    """
    return np.abs(((a - b + 180) % 360) - 180)


def safe_log10(arr, vmin=1e-30):
    """Safe log10 that handles zeros/negatives."""
    out = np.full_like(arr, np.nan, dtype=float)
    mask = arr > vmin
    out[mask] = np.log10(arr[mask])
    return out