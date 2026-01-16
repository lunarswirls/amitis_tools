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