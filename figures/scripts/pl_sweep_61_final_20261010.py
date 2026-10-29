#!/usr/bin/env python3
"""Figure 2: runs the merge and plotting scripts on figures/data/pl_sweep_20261009_61_fig_bbibm.json."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
run = lambda script, env: subprocess.check_call([sys.executable, os.path.join(HERE, script)], env=dict(os.environ, **env))
run("pl_sweep_61_bbibm_merge.py", {})
J = {"PL_FIG_JSON": "pl_sweep_20261009_61_fig_bbibm.json"}
run("pl_sweep_61_ab_v.py", dict(J, PL_P_OBS="0.002", PL_REF_ON_TOP="1"))
