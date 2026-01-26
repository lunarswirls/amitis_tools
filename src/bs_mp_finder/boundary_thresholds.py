#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:

"""
All thresholds are percentile of max change

At bowshock:
    - |V| strongly decreases
    - |J| moderately increases
    - |P| moderately increases

At magnetopause:
    - |J| significantly decreases
    - |P| moderately decreases
    - |B| significantly increases
"""

THRESHOLDS = {
    # bowshock threshold
    "Bgradmax_bs": 0.20,
    "Vgradmax_bs": 0.15,
    "Pgradmax_bs": 0.10,
    "Jgradmax_bs": 0.10,
    "rotmax_bs":   0.10,

    # magnetopause thresholds
    "Bgradmax_mp": 0.25,
    "Vgradmax_mp": 0.05,
    "Pgradmax_mp":  0.10,
    "Jgradmax_mp":  0.10,
    "rotmax_mp":    0.10,
}
