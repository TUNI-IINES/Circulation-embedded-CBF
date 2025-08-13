# Circulation-embedded-CBF-A-solution-to-avoid-undesired-equilibria-and-dysfunctional-circulation
This repository provides a simulation reproducing the experiments from https://doi.org/10.1016/j.robot.2025.105132, including the simulator, source code, and data.  Contains two scenarios, nonconvex obstacle avoidance and multi-agent deadlock resolution using IBCECBFQP, each runnable via lidar_gp_cbf/sim2D_main.

# Circulation-Embedded CBF — Code & Simulations

Code and simulation assets for the paper:

> **Circulation-embedded CBF: A solution to avoid undesired equilibria and dysfunctional circulation**  
> Shaghayegh Keyumarsi et al.

This repository provides two self-contained simulation packages and their produced media (figures/animations). A citable, permanent DOI will be added after we archive a release on Zenodo.

---

## Repository layout

- `nonconvex-sim/` — Nonconvex obstacle simulation + scripts  
  - main entry point: `nonconvex-sim/lidar_gp_cbf/simulator/sim2D_main.py`
- `multiagent-deadlock-sim/` — Multi-agent deadlock resolution simulation + scripts  
  - main entry point: `multiagent-deadlock-sim/lidar_gp_cbf/simulator/sim2D_main.py`
- Example outputs: images/videos are included alongside the code (e.g., `simulation ... .mp4`, plots, and any `animation_result/` subfolders).

> **Coming soon:** DOI badge + `CITATION.cff` after the first GitHub Release / Zenodo archive.

---

## Quick start

```bash
# 1) clone
git clone https://github.com/sh-keyumarsi/Circulation-embedded-CBF-A-solution-to-avoid-undesired-equilibria-and-dysfunctional-circulation.git
cd Circulation-embedded-CBF-A-solution-to-avoid-undesired-equilibria-and-dysfunctional-circulation

# 2) (recommended) create an isolated Python env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) install dependencies
# If a requirements.txt is present:
#   pip install -r requirements.txt
# Otherwise, install the minimal scientific stack you use (example):
#   pip install numpy scipy matplotlib
