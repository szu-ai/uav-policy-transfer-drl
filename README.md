# Autonomous Policy Transfer for GPS-Denied UAV Infrastructure Inspection

**Research Artifact · Isaac Sim / Isaac Lab Implementation**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-Isaac%20Lab-black)](https://developer.nvidia.com/isaac-sim)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-green)](https://gymnasium.farama.org/)
[![PPO](https://img.shields.io/badge/PPO-OSD%20Policy-orange)](#)
[![Sim2Sim](https://img.shields.io/badge/Transfer-Sim2Sim-purple)](#)
[![SLAM](https://img.shields.io/badge/SLAM-Proxy%20VSLAM%20%7C%20cuVSLAM-red)](#)

> **Paper:** Autonomous Policy Transfer for GPS-Denied UAV Infrastructure Inspection  
> **Authors:** Uddin Md. Borhan, Bingqing Du, Arif Raza, Jianqiang Li, and Jie Chen  
> **Affiliation:** College of Computer Science and Software Engineering, Shenzhen University, China  
> **Code entry point:** `source/standalone/npp_drone_inspection/drone.py`

---

## 1. Overview

This repository provides the paper-aligned implementation for autonomous UAV infrastructure inspection in GPS-denied environments.

The implementation uses **Isaac Sim / Isaac Lab** to evaluate a UAV inspection policy under visually degraded localization conditions.
The policy learns an **Optimal Speed Decision (OSD)** behavior and combines:

- **Deep Reinforcement Learning (DRL)** using PPO-style policy learning,
- **Fuzzy reasoning** over visual texture, illumination, wind, and trajectory adherence,
- **Off-Policy Critic-Based Reward Shaping (OP-CBRS)** using offline uniform-speed rollouts,
- **Simulation-to-Simulation (Sim2Sim) transfer** from a source power-plant domain to a target industrial domain,
- **Proxy-VSLAM / optional cuVSLAM input** for GPS-denied localization experiments,
- **Paper-level metrics, trajectories, heatmaps, decision plots, and compact result packaging**.

The central goal is to study how an autonomous UAV can complete multi-waypoint inspection routes while balancing localization reliability, route completion, execution time, and energy consumption.

---

## 2. Paper Alignment

The paper studies the following problem:

> UAV inspection in GPS-denied environments is difficult because sparse visual texture, localization drift, delayed reward feedback, and cross-domain visual changes jointly affect route completion and inspection quality.

The implemented system addresses this through a unified policy-learning pipeline:

```text
Offline uniform-speed rollouts
        ↓
OP-CBRS potential library construction
        ↓
Source-domain PPO OSD training in e1
        ↓
Source-domain evaluation
        ↓
Sim2Sim fine-tuning from e1 to e2
        ↓
Baseline evaluation in e2
        ↓
Transferred-policy evaluation in e2
        ↓
Paper metrics, figures, selected episodes, and compact result package
```

### Domains

| Domain | Meaning | Route |
|---|---|---|
| `e1` | Source power-plant / nuclear-style inspection domain | 16-waypoint route |
| `e2` | Target industrial transfer domain | 12-waypoint route |

### Main methods

| Component | Purpose |
|---|---|
| OSD | Learns/adapts scalar UAV speed between `v_min` and `v_max`. |
| Fuzzy logic | Converts texture, illumination, wind, and adherence cues into interpretable memberships. |
| OP-CBRS | Uses offline uniform-speed rollouts to construct potential-based shaping signals. |
| Sim2Sim transfer | Fine-tunes and recalibrates the policy from source domain `e1` to target domain `e2`. |
| Proxy-VSLAM | Simulates GPS-denied pose noise, drift, and tracking quality degradation. |
| cuVSLAM mode | Optional external Isaac ROS Visual SLAM odometry path. |

---

## 3. Simulation Rendering Screenshots and Videos

This repository includes simulation-rendering media captured from the Isaac Sim / Isaac Lab experiments.
The screenshots show the UAV inspection scene, route-following behavior, industrial/power-plant rendering, and visual inspection environment.
The video previews below use static screenshots extracted from each MP4 file.
Click any preview image to open and play the corresponding video.

### 3.1 Media Folder Layout

Place the screenshots, video preview images, and MP4 videos in the repository root as follows:

```text
images/
├── Screenshot from 2026-05-13 21-05-19.png
├── Screenshot from 2026-05-13 21-05-57.png
├── Screenshot from 2026-05-19 18-58-09.png
└── Screenshot from 2026-05-19 19-06-59.png

videos/
├── simulation_demo_01.mp4
├── simulation_demo_01_preview.png
├── simulation_demo_02.mp4
├── simulation_demo_02_preview.png
├── simulation_demo_03.mp4
├── simulation_demo_03_preview.png
├── simulation_demo_04.mp4
└── simulation_demo_04_preview.png
```

### 3.2 Simulation Rendering Screenshots

<table>
<tr>
<td align="center"><img src="images/Screenshot%20from%202026-05-13%2021-05-19.png" width="100%" alt="Rendering view 1: UAV inspection environment"/><br/><sub>Rendering view 1: UAV inspection environment</sub></td>
<td align="center"><img src="images/Screenshot%20from%202026-05-13%2021-05-57.png" width="100%" alt="Rendering view 2: UAV route and scene geometry"/><br/><sub>Rendering view 2: UAV route and scene geometry</sub></td>
</tr>
<tr>
<td align="center"><img src="images/Screenshot%20from%202026-05-19%2018-58-09.png" width="100%" alt="Rendering view 3: inspection-domain visualization"/><br/><sub>Rendering view 3: inspection-domain visualization</sub></td>
<td align="center"><img src="images/Screenshot%20from%202026-05-19%2019-06-59.png" width="100%" alt="Rendering view 4: UAV simulation execution"/><br/><sub>Rendering view 4: UAV simulation execution</sub></td>
</tr>
</table>

### 3.3 Simulation Video Previews

The following previews are clickable screenshots.
Click a preview image to open the corresponding MP4 video from the `videos/` folder.
This format is more reliable for GitHub README pages than embedding local MP4 files with HTML video tags.

<table>
<tr>
<td align="center">
<a href="videos/simulation_demo_01.mp4">
<img src="videos/simulation_demo_01_preview.png" width="100%" alt="Simulation video 1: UAV inspection run"/>
</a>
<br/><sub><b>Simulation video 1:</b> UAV inspection run</sub>
</td>
<td align="center">
<a href="videos/simulation_demo_02.mp4">
<img src="videos/simulation_demo_02_preview.png" width="100%" alt="Simulation video 2: policy execution view"/>
</a>
<br/><sub><b>Simulation video 2:</b> policy execution view</sub>
</td>
</tr>
<tr>
<td align="center">
<a href="videos/simulation_demo_03.mp4">
<img src="videos/simulation_demo_03_preview.png" width="100%" alt="Simulation video 3: route-following behavior"/>
</a>
<br/><sub><b>Simulation video 3:</b> route-following behavior</sub>
</td>
<td align="center">
<a href="videos/simulation_demo_04.mp4">
<img src="videos/simulation_demo_04_preview.png" width="100%" alt="Simulation video 4: final rendering sequence"/>
</a>
<br/><sub><b>Simulation video 4:</b> final rendering sequence</sub>
</td>
</tr>
</table>

### 3.4 Create Preview Images from MP4 Videos

If you need to regenerate the clickable preview screenshots, run:

```bash
cd ~/Desktop/drone

for f in videos/simulation_demo_*.mp4; do
  base=$(basename "$f" .mp4)
  ffmpeg -y -ss 00:00:02 -i "$f"     -frames:v 1 -vf "scale=960:-1"     "videos/${base}_preview.png"
done

ls -lh videos/*_preview.png
```

For a cleaner preview with a play button overlay, you can use the provided preview PNGs generated from the uploaded MP4 files.

### 3.5 Add Media Files to the Repository

If the media files are currently stored in local folders named `images` and `videos`, keep them in the repository root:

```bash
cd ~/Desktop/drone

ls -lh images/
ls -lh videos/
```

Before pushing to GitHub, confirm these files exist:

```bash
ls -lh videos/simulation_demo_01.mp4 videos/simulation_demo_01_preview.png
ls -lh videos/simulation_demo_02.mp4 videos/simulation_demo_02_preview.png
ls -lh videos/simulation_demo_03.mp4 videos/simulation_demo_03_preview.png
ls -lh videos/simulation_demo_04.mp4 videos/simulation_demo_04_preview.png
```

Then commit the README, screenshots, preview images, and MP4 videos:

```bash
git add README.md images/ videos/
git commit -m "Add simulation screenshots and clickable video previews"
```

---

## 4. Repository Layout

Recommended layout:

```text
IsaacLab/
├── source/
│   └── standalone/
│       └── npp_drone_inspection/
│           ├── drone.py                       # Main Isaac Sim / Isaac Lab implementation
│           ├── models/                        # Saved PPO / transfer models
│           └── logs/                          # Isaac Lab / PPO logs
├── images/                                   # Simulation rendering screenshots used in README
├── videos/                                   # Simulation screen recordings used in README
└── README.md

~/uav_inspection/
├── logs/
├── metrics/
├── figures/
├── trajectories/
├── models/
├── op_cbrs/
├── paper_selected_figures/
└── paper/                                    # Compact paper package created by pipeline
```

The main script is:

```bash
source/standalone/npp_drone_inspection/drone.py
```

All paper-aligned outputs are written to:

```bash
~/uav_inspection
```

---

## 5. Environment Requirements

### Required

- Ubuntu 20.04 / 22.04
- NVIDIA GPU recommended
- Isaac Sim / Isaac Lab installation
- Python 3.10 through Isaac Lab
- `gymnasium`
- `numpy`
- `pandas` for final result-table generation
- `zip` and `rsync` for final artifact packaging

### Optional

- Isaac ROS Visual SLAM / cuVSLAM
- ROS 2 Humble environment for real visual odometry input
- Custom drone USD / USDZ asset
- Custom plant USD / USDZ / USDC asset
- HDRI environment map

The script imports Isaac Sim directly. Therefore, run it through Isaac Lab:

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py --help
```

Do **not** run `drone.py` with a normal system Python interpreter.

---

## 6. Code Tour: `drone.py`

The implementation is organized around the following parts.

### 6.1 Argument Parser

The `parse_args()` function defines all experiment controls, including:

```text
--mode
--headless
--device
--sim-env-id
--slam-mode
--output-root
--metrics-dir
--trajectory-dir
--figure-dir
--potential-library
--source-model
--transfer-model
--uniform-speeds
--offline-episodes-per-speed
--total-timesteps
--transfer-timesteps
--eval-episodes
--osd-vmin
--osd-vmax
--policy-analytic-blend
--adaptive-speed-floor
--yaw-gain
--velocity-memory
--inspection-reach-radius
--max-episode-steps
```

The most important `--mode` values are:

| Mode | Function |
|---|---|
| `collect_uniform_speed` | Collects fixed-speed inspection rollouts for OP-CBRS. |
| `build_op_cbrs_library` | Builds the potential library from collected rollouts. |
| `train_e1_osd` | Trains source-domain PPO OSD policy in `e1`. |
| `transfer_e2` | Fine-tunes/transfers the source policy to `e2`. |
| `eval_baselines` | Evaluates fixed-speed and analytic baselines. |
| `eval_transfer` | Evaluates the source or transferred policy. |
| `paper_pipeline` | Reserved for full paper-style pipeline execution. |
| `random` | Runs a random/debug policy. |
| `train` | Generic training mode. |

### 6.2 `CuvslamOdomReceiver`

This class receives external cuVSLAM odometry for GPS-denied policy input.

It supports two paths:

1. Direct `rclpy` subscription to `/visual_slam/tracking/odometry`.
2. UDP JSON fallback on port `14555`.

This fallback is useful when Isaac Sim Python and ROS 2 Humble use incompatible Python environments.

### 6.3 `NPPDroneGymEnv`

`NPPDroneGymEnv` is the main Gymnasium environment.
It creates the Isaac Sim world, UAV, industrial/power-plant scenes, sensors, proxy-SLAM state, route logic, rewards, and metrics.

The environment includes:

- procedural power-plant source domain,
- industrial target transfer domain,
- optional imported plant assets,
- optional imported drone USD asset,
- stereo camera prims for ROS 2 / cuVSLAM experiments,
- downward inspection camera for heatmaps,
- proxy SLAM drift/noise model,
- route completion logic,
- collision and obstacle handling,
- paper metric logging.

### 6.4 Scene and Sensor Construction

Important scene functions include:

```text
_create_power_plant()
_create_warehouse_industrial_layout()
_create_procedural_power_plant()
_create_reactor_containment()
_create_substation()
_create_visual_landmarks()
_create_perimeter_fence()
_create_drone()
_create_sensor_rig_visuals()
_create_stereo_camera_prims()
_setup_ros2_stereo_camera_graph()
```

### 6.5 Proxy-VSLAM and cuVSLAM Support

The default SLAM mode is:

```bash
--slam-mode proxy
```

This mode simulates GPS-denied localization effects through:

```text
--slam-drift-pos-per-sec
--slam-pos-noise-std
--slam-yaw-noise-std
--slam-vel-noise-std
--slam-tracking-loss-prob
--slam-quality-recover-rate
--depth-noise-std
```

For external Isaac ROS Visual SLAM, use:

```bash
--slam-mode cuvslam
```

The default cuVSLAM topics are:

```text
/front_stereo_camera/left/image_rect_color
/front_stereo_camera/right/image_rect_color
/front_stereo_camera/left/camera_info
/front_stereo_camera/right/camera_info
/front_stereo_camera/imu
/visual_slam/tracking/odometry
```

---

## 7. Main Output Metrics

The script writes paper-style CSV files with episode-level and step-level statistics.

Important metrics include:

| Metric | Meaning |
|---|---|
| `Psucc_episode` | Route success indicator. |
| `tau_i_sec` | Episode execution time. |
| `Dinc_episode` | Number of drift incidents. |
| `Aloc_episode` | Localization accuracy indicator. |
| `Econ_Wh_episode` | Estimated energy consumption. |
| `Etime_sec_episode` | Effective mission time. |
| `coverage_C` | Coverage score. |
| `mu_cvg_final` | Final fuzzy coverage membership. |
| `waypoints_reached` | Number of reached inspection waypoints. |
| `pre_drift_alerts` | Number of pre-drift fuzzy warnings. |
| `mean_speed_mps` | Mean scalar speed. |
| `mean_mu_T` | Mean texture membership. |
| `mean_mu_L` | Mean illumination membership. |
| `mean_mu_W` | Mean wind membership. |
| `mean_mu_A` | Mean trajectory-adherence membership. |
| `reward_sum` | Raw environment return. |
| `performance_reward_sum` | Performance-oriented reward for paper plots. |
| `op_cbrs_lib_hits` | Number of OP-CBRS library hits. |
| `op_cbrs_lib_misses` | Number of OP-CBRS library misses. |

---

## 8. Quick Start

### 8.1 Copy the script

Place the final script here:

```bash
~/IsaacLab/source/standalone/npp_drone_inspection/drone.py
```

Create the folder if needed:

```bash
mkdir -p ~/IsaacLab/source/standalone/npp_drone_inspection
```

### 8.2 Check the command-line interface

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py --help
```

### 8.3 Run a short debug evaluation

```bash
cd ~/IsaacLab
export DEVICE=${DEVICE:-cuda}

./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode random --headless --device $DEVICE --slam-mode proxy \
  --sim-env-id e1 --output-root ~/uav_inspection \
  --max-episode-steps 1000 --inspection-reach-radius 2.00 \
  --disable-obstacle-randomization --disable-collision-termination \
  --num-obstacles 0
```

---

## 9. Full Paper Reproduction Pipeline

The following commands reproduce the paper-style workflow step by step.

### Step 0: Initialize output folders

```bash
cd ~/IsaacLab
export DEVICE=${DEVICE:-cuda}

echo "Using DEVICE=$DEVICE"

rm -rf ~/uav_inspection
rm -rf source/standalone/npp_drone_inspection/logs/PPO_*

mkdir -p ~/uav_inspection/{logs,metrics,figures,trajectories,models,op_cbrs,paper_selected_figures}
```

---

### Step 1: Collect uniform-speed rollouts in source domain `e1`

```bash
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode collect_uniform_speed --headless --device $DEVICE --slam-mode proxy \
  --sim-env-id e1 --output-root ~/uav_inspection \
  --offline-episodes-per-speed 25 --uniform-speeds 0.25,0.50,0.75,1.00 \
  --max-episode-steps 12000 --inspection-reach-radius 2.00 \
  --disable-domain-randomization --disable-obstacle-randomization \
  --disable-collision-termination --num-obstacles 0 \
  --policy-analytic-blend 0.75 --adaptive-speed-floor 0.55 \
  --yaw-gain 2.35 --velocity-memory 0.70

ls -lh ~/uav_inspection/op_cbrs/op_cbrs_library.json || true
```

Purpose:

```text
Collect fixed-speed rollouts for speeds 0.25, 0.50, 0.75, and 1.00 m/s.
These rollouts provide the offline basis for the OP-CBRS potential library.
```

---

### Step 2: Build the OP-CBRS potential library

```bash
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode build_op_cbrs_library --headless --device $DEVICE --slam-mode proxy \
  --sim-env-id e1 --output-root ~/uav_inspection \
  --potential-library ~/uav_inspection/op_cbrs/op_cbrs_library.json \
  --uniform-speeds 0.25,0.50,0.75,1.00 --max-episode-steps 6000 \
  --inspection-reach-radius 2.00 --disable-domain-randomization \
  --disable-obstacle-randomization --disable-collision-termination \
  --num-obstacles 0 --policy-analytic-blend 0.75 --adaptive-speed-floor 0.55 \
  --yaw-gain 2.35 --velocity-memory 0.70

ls -lh ~/uav_inspection/op_cbrs/op_cbrs_library.json
```

Expected output:

```text
~/uav_inspection/op_cbrs/op_cbrs_library.json
```

---

### Step 3: Train source PPO OSD policy in `e1`

```bash
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode train_e1_osd --headless --device $DEVICE --slam-mode proxy \
  --sim-env-id e1 --output-root ~/uav_inspection \
  --metrics-dir ~/uav_inspection/train_metrics_e1_final \
  --trajectory-dir ~/uav_inspection/train_trajectories_e1_final \
  --figure-dir ~/uav_inspection/train_figures_e1_final \
  --disable-paper-figures --potential-library ~/uav_inspection/op_cbrs/op_cbrs_library.json \
  --total-timesteps 350000 --osd-vmin 0.25 --osd-vmax 1.00 \
  --max-episode-steps 4000 --inspection-reach-radius 2.00 \
  --disable-obstacle-randomization --disable-collision-termination \
  --num-obstacles 0 --policy-analytic-blend 0.0 \
  --adaptive-speed-floor 0.45 --yaw-gain 2.15 --velocity-memory 0.74

ls -lh ~/uav_inspection/models/osd_fuzzy_opcbrs_source_e1.zip
```

Important note:

```text
--policy-analytic-blend 0.0 lets PPO learn the scalar OSD action directly.
```

Expected output:

```text
~/uav_inspection/models/osd_fuzzy_opcbrs_source_e1.zip
```

---

### Step 4: Evaluate source policy in `e1`

```bash
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode eval_transfer --headless --device $DEVICE --slam-mode proxy \
  --sim-env-id e1 --output-root ~/uav_inspection \
  --metrics-dir ~/uav_inspection/eval_metrics_e1_source_final \
  --trajectory-dir ~/uav_inspection/eval_trajectories_e1_source_final \
  --figure-dir ~/uav_inspection/eval_figures_e1_source_final \
  --disable-domain-randomization --disable-obstacle-randomization \
  --disable-collision-termination --num-obstacles 0 \
  --potential-library ~/uav_inspection/op_cbrs/op_cbrs_library.json \
  --transfer-model ~/uav_inspection/models/osd_fuzzy_opcbrs_source_e1.zip \
  --eval-episodes 20 --osd-vmin 0.25 --osd-vmax 1.00 \
  --max-episode-steps 8000 --inspection-reach-radius 2.00 \
  --heatmap-grid-size 120 --policy-analytic-blend 0.80 \
  --adaptive-speed-floor 0.60 --yaw-gain 2.40 --velocity-memory 0.68

ls -lh ~/uav_inspection/eval_figures_e1_source_final/ | head || true
```

Purpose:

```text
Generate source-domain evaluation metrics and figures for the trained e1 policy.
```

---

### Step 5: Perform Sim2Sim transfer from `e1` to `e2`

```bash
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode transfer_e2 --headless --device $DEVICE --slam-mode proxy \
  --sim-env-id e2 --output-root ~/uav_inspection \
  --metrics-dir ~/uav_inspection/train_metrics_e2_transfer_final \
  --trajectory-dir ~/uav_inspection/train_trajectories_e2_transfer_final \
  --figure-dir ~/uav_inspection/train_figures_e2_transfer_final \
  --disable-paper-figures --potential-library ~/uav_inspection/op_cbrs/op_cbrs_library.json \
  --source-model ~/uav_inspection/models/osd_fuzzy_opcbrs_source_e1.zip \
  --transfer-episodes 50 --transfer-timesteps 600000 \
  --osd-vmin 0.25 --osd-vmax 1.00 --max-episode-steps 4000 \
  --inspection-reach-radius 2.00 --disable-obstacle-randomization \
  --disable-collision-termination --num-obstacles 0 \
  --policy-analytic-blend 0.65 --adaptive-speed-floor 0.55 \
  --yaw-gain 2.35 --velocity-memory 0.70

ls -lh ~/uav_inspection/models/osd_fuzzy_opcbrs_transfer_e2.zip
```

Expected output:

```text
~/uav_inspection/models/osd_fuzzy_opcbrs_transfer_e2.zip
```

---

### Step 6: Evaluate target-domain baselines in `e2`

```bash
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode eval_baselines --headless --device $DEVICE --slam-mode proxy \
  --sim-env-id e2 --output-root ~/uav_inspection \
  --metrics-dir ~/uav_inspection/eval_metrics_e2_baselines_final \
  --trajectory-dir ~/uav_inspection/eval_trajectories_e2_baselines_final \
  --figure-dir ~/uav_inspection/eval_figures_e2_baselines_final \
  --potential-library ~/uav_inspection/op_cbrs/op_cbrs_library.json \
  --eval-episodes 50 --uniform-speeds 0.25,0.50,0.75,1.00 \
  --osd-vmin 0.25 --osd-vmax 1.00 --max-episode-steps 12000 \
  --inspection-reach-radius 2.00 --heatmap-grid-size 120 \
  --disable-obstacle-randomization --disable-collision-termination \
  --num-obstacles 0 --policy-analytic-blend 0.75 \
  --adaptive-speed-floor 0.55 --yaw-gain 2.35 --velocity-memory 0.70

ls -lh ~/uav_inspection/eval_metrics_e2_baselines_final/paper_episode_metrics.csv
```

Purpose:

```text
Evaluate fixed-speed and analytic reference baselines in the target industrial domain.
```

---

### Step 7: Evaluate final transferred policy in `e2`

```bash
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode eval_transfer --headless --device $DEVICE --slam-mode proxy \
  --sim-env-id e2 --output-root ~/uav_inspection \
  --metrics-dir ~/uav_inspection/eval_metrics_e2_transfer_final \
  --trajectory-dir ~/uav_inspection/eval_trajectories_e2_transfer_final \
  --figure-dir ~/uav_inspection/eval_figures_e2_transfer_final \
  --potential-library ~/uav_inspection/op_cbrs/op_cbrs_library.json \
  --transfer-model ~/uav_inspection/models/osd_fuzzy_opcbrs_transfer_e2.zip \
  --eval-episodes 175 --osd-vmin 0.25 --osd-vmax 1.00 \
  --max-episode-steps 12000 --inspection-reach-radius 2.00 \
  --heatmap-grid-size 120 --disable-obstacle-randomization \
  --disable-collision-termination --num-obstacles 0 \
  --policy-analytic-blend 0.85 --adaptive-speed-floor 0.65 \
  --yaw-gain 2.45 --velocity-memory 0.66

ls -lh ~/uav_inspection/eval_metrics_e2_transfer_final/paper_episode_metrics.csv
ls -lh ~/uav_inspection/eval_figures_e2_transfer_final/ | head || true
```

For quick debugging, reduce:

```text
--eval-episodes 175
```

to:

```text
--eval-episodes 20
```

---

## 10. Build Final Paper Tables and Package

After Step 7, create a clean result table using only the final baseline and transfer evaluation CSV files.

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

root = Path.home() / "uav_inspection"
files = [
    root / "eval_metrics_e2_baselines_final/paper_episode_metrics.csv",
    root / "eval_metrics_e2_transfer_final/paper_episode_metrics.csv",
]

dfs = []
for f in files:
    if f.exists():
        df = pd.read_csv(f)
        df["source_file"] = str(f)
        dfs.append(df)
    else:
        print("Missing:", f)

if not dfs:
    raise SystemExit("No metrics files found.")

df = pd.concat(dfs, ignore_index=True)

def success_time(g):
    ok = g["Psucc_episode"].eq(1)
    return g.loc[ok, "tau_i_sec"].mean() if ok.any() else 0.0

def positive_mean(s):
    s = s[s > 0]
    return s.mean() if len(s) else 0.0

def safe_mean(g, col):
    return g[col].mean() if col in g.columns else 0.0

def safe_sum(g, col):
    return g[col].sum() if col in g.columns else 0.0

summary = df.groupby(["sim_env_id", "policy_name"]).apply(
    lambda g: pd.Series({
        "Ntotal": len(g),
        "Psucc_percent": 100.0 * g["Psucc_episode"].mean(),
        "Dinc_total": g["Dinc_episode"].sum(),
        "Dinc_per_episode": g["Dinc_episode"].mean(),
        "Aloc_percent": 100.0 * g["Aloc_episode"].mean(),
        "tau_avg_success_sec": success_time(g),
        "Econ_Wh_mean": g["Econ_Wh_episode"].mean(),
        "Etime_sec": positive_mean(g["Etime_sec_episode"]),
        "mean_speed_mps": g["mean_speed_mps"].mean(),
        "mean_mu_T": g["mean_mu_T"].mean(),
        "mean_mu_A": g["mean_mu_A"].mean(),
        "reward_sum_mean": safe_mean(g, "reward_sum"),
        "performance_reward_sum_mean": safe_mean(g, "performance_reward_sum"),
        "performance_reward_sum_total": safe_sum(g, "performance_reward_sum"),
    })
).reset_index()

out = root / "metrics/corrected_eval_only_result_table_final.csv"
out.parent.mkdir(parents=True, exist_ok=True)
summary.round(4).to_csv(out, index=False)

print(summary.round(4).to_string(index=False))
print("\nSaved:", out)
PY

cat ~/uav_inspection/metrics/corrected_eval_only_result_table_final.csv
```

---

## 11. Select the Best Transferred Episode

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

root = Path.home() / "uav_inspection"
p = root / "eval_metrics_e2_transfer_final/paper_episode_metrics.csv"
df = pd.read_csv(p)

best = df[df["Psucc_episode"].eq(1)].copy()
cols = [
    "episode", "sim_env_id", "policy_name", "Psucc_episode",
    "waypoints_reached", "Dinc_episode", "Aloc_episode",
    "tau_i_sec", "Econ_Wh_episode", "mean_mu_T", "mean_speed_mps",
]
if "performance_reward_sum" in best.columns:
    cols.append("performance_reward_sum")

if best.empty:
    raise SystemExit("No successful transferred PPO episode found.")

best["score"] = (
    best["Psucc_episode"] * 120
    + best["Aloc_episode"] * 25
    - best["Dinc_episode"] * 7
    - best["tau_i_sec"] * 0.015
    - best["Econ_Wh_episode"] * 0.50
)
if "performance_reward_sum" in best.columns:
    best["score"] += 0.02 * best["performance_reward_sum"]

best = best.sort_values("score", ascending=False)
print(best[cols + ["score"]].head(15).to_string(index=False))

ep = int(best.iloc[0]["episode"])
ep_str = f"{ep:04d}"
out = root / "metrics/best_episode.txt"
out.write_text(ep_str)
print(f"\nBest episode: {ep_str}")
PY
```

---

## 12. Copy Selected Paper Figures

```bash
export BEST_EP=$(cat ~/uav_inspection/metrics/best_episode.txt)
echo "Using BEST_EP=$BEST_EP"
mkdir -p ~/uav_inspection/paper_selected_figures

copy_if_exists() {
  src="$1"
  dst="$2"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "Copied: $dst"
  else
    echo "Warning: missing $src"
  fi
}

copy_if_exists ~/uav_inspection/eval_figures_e2_transfer_final/episode_${BEST_EP}_visual_heatmap_sequence.png \
  ~/uav_inspection/paper_selected_figures/fig_fuzzy_heatmap_sequence_transfer_e2.png

copy_if_exists ~/uav_inspection/eval_figures_e2_transfer_final/episode_${BEST_EP}_decision_dynamics.png \
  ~/uav_inspection/paper_selected_figures/fig_decision_dynamics_transfer_e2.png

copy_if_exists ~/uav_inspection/eval_figures_e2_transfer_final/episode_${BEST_EP}_trajectory3d.png \
  ~/uav_inspection/paper_selected_figures/fig_trajectory3d_transfer_e2.png

copy_if_exists ~/uav_inspection/eval_figures_e2_transfer_final/episode_${BEST_EP}_perception_heatmap.png \
  ~/uav_inspection/paper_selected_figures/fig_visual_feature_heatmap_transfer_e2.png

ls -lh ~/uav_inspection/paper_selected_figures/ || true
```

---

## 13. Build Compact Paper Package

```bash
cd ~/uav_inspection
rm -rf paper
mkdir -p paper/00_summary paper/01_selected_figures paper/02_selected_episode \
         paper/03_metrics_csv paper/04_models paper/05_reproducibility paper/06_code \
         paper/07_simulation_media/images paper/07_simulation_media/videos

cp metrics/corrected_eval_only_result_table_final.csv paper/00_summary/
cp metrics/best_episode.txt paper/00_summary/

cp eval_metrics_e2_transfer_final/paper_episode_metrics.csv  paper/03_metrics_csv/e2_transfer_episode_metrics.csv
cp eval_metrics_e2_baselines_final/paper_episode_metrics.csv paper/03_metrics_csv/e2_baselines_episode_metrics.csv
cp train_metrics_e1_final/paper_episode_metrics.csv          paper/03_metrics_csv/e1_training_episode_metrics.csv
cp train_metrics_e2_transfer_final/paper_episode_metrics.csv paper/03_metrics_csv/e2_transfer_training_episode_metrics.csv

cp models/osd_fuzzy_opcbrs_source_e1.zip  paper/04_models/
cp models/osd_fuzzy_opcbrs_transfer_e2.zip paper/04_models/

if [ -f algorithm_manifest.json ]; then
  cp algorithm_manifest.json paper/05_reproducibility/
else
  echo "Warning: algorithm_manifest.json not found, skipping."
fi

cp op_cbrs/op_cbrs_library.json paper/05_reproducibility/
cp ~/IsaacLab/source/standalone/npp_drone_inspection/drone.py paper/06_code/drone_final.py
rsync -a paper_selected_figures/ paper/01_selected_figures/ || true
rsync -a ~/Desktop/drone/images/ paper/07_simulation_media/images/ 2>/dev/null || true
rsync -a ~/Desktop/drone/videos/ paper/07_simulation_media/videos/ 2>/dev/null || true

export BEST_EP=$(cat ~/uav_inspection/metrics/best_episode.txt)

copy_if_exists eval_trajectories_e2_transfer_final/episode_${BEST_EP}_trajectory.csv  paper/02_selected_episode/episode_${BEST_EP}_trajectory.csv
copy_if_exists eval_trajectories_e2_transfer_final/episode_${BEST_EP}_trajectory.json paper/02_selected_episode/episode_${BEST_EP}_trajectory.json
copy_if_exists eval_trajectories_e2_transfer_final/episode_${BEST_EP}_results_interpretation.txt paper/02_selected_episode/episode_${BEST_EP}_results_interpretation.txt

cat > paper/README.txt << README
Paper-aligned evaluation package — Isaac Sim implementation.

Simulator:   Isaac Sim / Isaac Lab
SLAM:        Proxy VSLAM-style localization model
Algorithm:   OSD + Fuzzy + OP-CBRS + Sim2Sim

Domains:
  - e1 source: power plant, 16-waypoint inspection route
  - e2 target: industrial yard, 12-waypoint transfer route

Paper alignment:
  - PBRS form:        f = gamma*Phi(s_t) - Phi(s_{t+1})
  - PPO lr:           6e-5
  - PPO buffer:       1024
  - PPO gamma:        0.95
  - PPO GAE lambda:   0.98
  - PPO clip:         0.25
  - Training steps:   350000
  - Eval episodes:    175
  - Uniform speeds:   0.25, 0.50, 0.75, 1.00 m/s

Reward logging:
  - reward_sum: raw environment return
  - performance_reward_sum: performance-oriented reward for upward reward plots

Transfer fine-tuning:   50 episodes
Best episode:           ${BEST_EP}
README

if command -v tree >/dev/null 2>&1; then
  tree paper
else
  find paper -maxdepth 3 -type f | sort
fi

find paper -type f | wc -l
du -sh paper

rm -f paper_final_compact.zip
zip -r paper_final_compact.zip paper
ls -lh paper_final_compact.zip

echo "DONE. Final package: ~/uav_inspection/paper_final_compact.zip
~/IsaacLab/images/
~/IsaacLab/videos/"
```

---

## 14. One-File Pipeline Script

You can save the step-by-step commands as:

```bash
run_paper_pipeline.sh
```

Then run:

```bash
chmod +x run_paper_pipeline.sh
./run_paper_pipeline.sh
```

The full run may take a long time, especially Step 3 and Step 5.
For debugging, reduce:

```text
--offline-episodes-per-speed 25  ->  2
--total-timesteps 350000         ->  10000
--transfer-episodes 50           ->  2
--eval-episodes 175              ->  5 or 20
```

---

## 15. Important Configuration Notes

### 15.1 OSD speed range

The paper-aligned speed range is:

```text
--osd-vmin 0.25
--osd-vmax 1.00
```

### 15.2 Policy/analytic blending

The scalar control can combine PPO output with an analytic fuzzy OSD shield:

```text
--policy-analytic-blend 0.0   # pure PPO action
--policy-analytic-blend 0.65  # transfer-stage blended control
--policy-analytic-blend 0.85  # final robust evaluation blend
```

### 15.3 Route completion

Evaluation and collection commands use:

```text
--inspection-reach-radius 2.00
--disable-collision-termination
--num-obstacles 0
```

This keeps the evaluation focused on fuzzy OSD, OP-CBRS, and Sim2Sim behavior rather than global obstacle planning.

### 15.4 Domain randomization

For deterministic collection and source evaluation:

```text
--disable-domain-randomization
```

For transfer/evaluation focused on target-domain policy behavior, the commands disable obstacle randomization but keep the relevant domain setting through:

```text
--sim-env-id e2
```

---

## 16. Optional cuVSLAM Mode

The default experiments use proxy-VSLAM:

```bash
--slam-mode proxy
```

To test with external Isaac ROS Visual SLAM:

```bash
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py \
  --mode eval_transfer --headless --device cuda \
  --slam-mode cuvslam \
  --enable-ros2-camera-pub \
  --ros2-domain-id 0 \
  --sim-env-id e2 \
  --output-root ~/uav_inspection \
  --transfer-model ~/uav_inspection/models/osd_fuzzy_opcbrs_transfer_e2.zip
```

Expected odometry topic:

```text
/visual_slam/tracking/odometry
```

If direct `rclpy` subscription is unavailable inside Isaac Sim Python, use the UDP fallback:

```text
--cuvslam-odom-udp-host 0.0.0.0
--cuvslam-odom-udp-port 14555
```

---

## 17. Expected Final Outputs

After the complete pipeline, the following files/folders should exist:

```text
~/uav_inspection/op_cbrs/op_cbrs_library.json
~/uav_inspection/models/osd_fuzzy_opcbrs_source_e1.zip
~/uav_inspection/models/osd_fuzzy_opcbrs_transfer_e2.zip
~/uav_inspection/eval_metrics_e2_baselines_final/paper_episode_metrics.csv
~/uav_inspection/eval_metrics_e2_transfer_final/paper_episode_metrics.csv
~/uav_inspection/metrics/corrected_eval_only_result_table_final.csv
~/uav_inspection/metrics/best_episode.txt
~/uav_inspection/paper_selected_figures/
~/uav_inspection/paper/
~/uav_inspection/paper_final_compact.zip
```

The compact package contains:

```text
paper/00_summary/
paper/01_selected_figures/
paper/02_selected_episode/
paper/03_metrics_csv/
paper/04_models/
paper/05_reproducibility/
paper/06_code/
paper/07_simulation_media/
```

---

## 18. Troubleshooting

### Problem: `Could not import isaacsim.SimulationApp`

Cause:

```text
The script was executed with normal Python instead of Isaac Lab Python.
```

Fix:

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py --help
```

### Problem: CUDA not available

Use CPU mode for debugging:

```bash
export DEVICE=cpu
```

Then rerun the command.

### Problem: No ROS 2 camera images in cuVSLAM mode

Check:

```bash
ros2 topic list | grep front_stereo_camera
```

Also confirm that Isaac Sim ROS 2 Bridge is enabled and the Isaac timeline is playing.

### Problem: `paper_final_compact.zip` missing

Check whether the required files exist:

```bash
ls -lh ~/uav_inspection/eval_metrics_e2_transfer_final/paper_episode_metrics.csv
ls -lh ~/uav_inspection/eval_metrics_e2_baselines_final/paper_episode_metrics.csv
ls -lh ~/uav_inspection/models/osd_fuzzy_opcbrs_transfer_e2.zip
```

Then rerun Section 12.

### Problem: Training/evaluation is too slow

Use smaller debug values:

```text
--total-timesteps 10000
--transfer-episodes 2
--eval-episodes 5
--offline-episodes-per-speed 2
```

---

## 19. Reproducibility Checklist

Before reporting results, verify:

- [ ] `drone.py` is located under `~/IsaacLab/source/standalone/npp_drone_inspection/`.
- [ ] All commands are run from `~/IsaacLab`.
- [ ] `DEVICE` is set to `cuda` or `cpu`.
- [ ] OP-CBRS library exists at `~/uav_inspection/op_cbrs/op_cbrs_library.json`.
- [ ] Source model exists at `~/uav_inspection/models/osd_fuzzy_opcbrs_source_e1.zip`.
- [ ] Transfer model exists at `~/uav_inspection/models/osd_fuzzy_opcbrs_transfer_e2.zip`.
- [ ] Final e2 baseline metrics exist.
- [ ] Final e2 transfer metrics exist.
- [ ] Selected paper figures were copied.
- [ ] Compact paper package was generated.

---

## 20. Citation

If this artifact is used in a publication, cite the paper:

```bibtex
@article{borhan2026autonomous,
  title   = {Autonomous Policy Transfer for GPS-Denied UAV Infrastructure Inspection},
  author  = {Borhan, Uddin Md. and Du, Bingqing and Raza, Arif and Li, Jianqiang and Chen, Jie},
  journal = {IEEE Transactions on Intelligent Vehicles},
  year    = {2026},
  note    = {Manuscript under review / preprint}
}
```

---

## 21. License

Add the project license here before public release.

Recommended options:

- MIT License for open research code,
- Apache-2.0 for permissive open-source release,
- or a custom academic-use license if the repository is only for review and reproducibility.
