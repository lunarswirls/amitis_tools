#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import re
import numpy as np
import pandas as pd

# check that nx, ny, nz divide into total x, y, z extents
# check that number of GPUs is equal in input and bash script


def parse_input_variables(filename: str):
    """
    Parse a text file for lines like: variable = value
    - Strip anything after a '#'
    - Ignore lines that become empty or start with '#'
    - Return a dictionary {variable: value}
    """
    pattern = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
    variables = {}

    with open(filename, 'r') as f:
        for line in f:
            # Remove trailing comments
            stripped = line.split('#', 1)[0].strip()

            # Skip empty or comment-only lines
            if not stripped:
                continue

            match = pattern.match(stripped)
            if match:
                var, val = match.groups()
                variables[var] = val

    return variables


