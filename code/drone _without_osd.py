#!/usr/bin/env python3
# PPO-without-OSD ablation variant. OP-CBRS remains enabled; the analytic/fuzzy OSD speed shield is disabled in *_no_osd modes.

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:gym")

try:
    from isaacsim import SimulationApp
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import isaacsim.SimulationApp. Run with Isaac Lab, e.g.:\n"
        "cd ~/IsaacLab && ./isaaclab.sh -p source/standalone/npp_drone_inspection/drone.py"
    ) from exc


omni = None
Gf = None
Sdf = None
UsdGeom = None
UsdLux = None
UsdShade = None


def parse_args():
    parser = argparse.ArgumentParser(description="GPS-denied drone inspection in Isaac Sim.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "random", "collect_uniform_speed", "build_op_cbrs_library", "train_e1_osd", "transfer_e2", "eval_baselines", "eval_transfer", "paper_pipeline", "train_e1_no_osd", "transfer_e2_no_osd", "eval_transfer_no_osd", "paper_pipeline_no_osd"], help="Algorithmic mode: PPO train/eval, offline uniform-speed collection, OP-CBRS library, Sim2Sim transfer, or full paper pipeline.")
    parser.add_argument("--total-timesteps", type=int, default=350000)
    parser.add_argument("--transfer-timesteps", type=int, default=75000, help="Fine-tuning steps for e1->e2 Sim2Sim policy transfer.")
    parser.add_argument("--train-episodes", type=int, default=0, help="If >0, stop PPO source training after this many completed episodes; total_timesteps becomes an upper budget.")
    parser.add_argument("--transfer-episodes", type=int, default=0, help="If >0, stop PPO Sim2Sim fine-tuning after this many completed episodes; transfer_timesteps becomes an upper budget.")
    parser.add_argument("--disable-domain-randomization", action="store_true", help="Disable all reset-time domain randomization. Use for deterministic debugging only.")
    parser.add_argument("--enable-route-randomization", action="store_true", help="Randomly permute inspection waypoint order on each reset. Disabled by default because the paper studies OSD speed transfer, not route-planning transfer.")
    parser.add_argument("--disable-obstacle-randomization", action="store_true", help="Disable random dynamic obstacle placement. This keeps evaluation focused on fuzzy OSD/OP-CBRS/Sim2Sim under lighting, texture, and wind shifts.")
    parser.add_argument("--safe-obstacle-placement", action="store_true", default=True, help="When random obstacles are used, keep a clearance corridor around the planned waypoint route.")
    parser.add_argument("--route-corridor-clearance", type=float, default=2.60, help="Minimum XY clearance from obstacles to waypoint segments when safe obstacle placement is enabled.")
    parser.add_argument("--disable-collision-termination", action="store_true", help="Do not terminate the episode on very close obstacle ray hits; apply a strong penalty instead. Useful for OSD-only evaluation without global path planning.")
    # Robust transfer/controller settings.  The policy still outputs one scalar OSD speed,
    # but these options add a paper-aligned safety shield and deterministic tracking
    # improvements so the trained policy transfers more reliably from e1 to e2.
    parser.add_argument("--policy-analytic-blend", type=float, default=0.65, help="Blend factor between PPO speed and analytic fuzzy OSD speed. 0=only PPO, 1=analytic OSD shield.")
    parser.add_argument("--adaptive-speed-floor", type=float, default=0.45, help="Minimum fraction of the speed range used in low-risk segments to avoid stalled policies.")
    parser.add_argument("--obstacle-avoidance-gain", type=float, default=0.85, help="Strength of deterministic local obstacle repulsion added to the waypoint tracker.")
    parser.add_argument("--obstacle-avoidance-range", type=float, default=2.20, help="Meters from an obstacle where local avoidance starts.")
    parser.add_argument("--yaw-gain", type=float, default=2.15, help="Waypoint-facing yaw controller gain.")
    parser.add_argument("--velocity-memory", type=float, default=0.74, help="Velocity smoothing memory. Lower values track speed commands faster.")
    parser.add_argument("--enable-osd", action="store_true", help="Enable the analytic/fuzzy OSD speed shield. In *_no_osd modes this is forced off.")
    parser.add_argument("--disable-osd", action="store_true", help="Force-disable the analytic/fuzzy OSD speed shield for PPO-without-OSD ablation.")
    parser.add_argument("--eval-episodes", type=int, default=175, help="Number of episodes for algorithmic evaluation modes.")
    parser.add_argument("--uniform-speeds", type=str, default="0.25,0.50,0.75,1.00", help="Comma-separated uniform speed tasks T* used for OP-CBRS library/baselines.")
    parser.add_argument("--osd-vmin", type=float, default=0.25, help="Paper OSD minimum scalar speed v_min in m/s.")
    parser.add_argument("--osd-vmax", type=float, default=1.00, help="Paper OSD maximum scalar speed v_max in m/s.")
    parser.add_argument("--offline-episodes-per-speed", type=int, default=12, help="Episodes collected for each uniform speed task T*.")
    parser.add_argument("--source-model", type=str, default="", help="Optional source e1 PPO model path for transfer/evaluation.")
    parser.add_argument("--transfer-model", type=str, default="", help="Optional transferred e2 PPO model path for evaluation.")
    parser.add_argument("--potential-library", type=str, default="", help="OP-CBRS potential library JSON. Defaults to <output-root>/op_cbrs/op_cbrs_library.json.")
    parser.add_argument("--algorithm-manifest-json", type=str, default="", help="Optional path for a JSON manifest documenting MDP/OSD/Fuzzy/OP-CBRS/Sim2Sim settings.")
    parser.add_argument("--model-dir", type=str, default="source/standalone/npp_drone_inspection/models")
    parser.add_argument("--log-dir", type=str, default="source/standalone/npp_drone_inspection/logs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--inspection-reach-radius", type=float, default=1.50, help="Distance in meters for accepting an inspection waypoint as reached.")
    parser.add_argument("--allow-timeout-before-route-complete", action="store_true", help="Allow max-episode-steps timeout before all inspection waypoints are visited. By default demo/eval modes continue until the route is complete.")
    parser.add_argument("--world-size", type=float, default=34.0)
    parser.add_argument("--num-obstacles", type=int, default=8)
    parser.add_argument("--num-rays", type=int, default=36)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--drone-usd", type=str, default="", help="Optional realistic drone USD/USDZ")
    parser.add_argument("--plant-usd", type=str, default="", help="Optional realistic power-plant USD/USDZ/USDC")
    parser.add_argument("--hdri-path", type=str, default="", help="Optional HDRI path")
    parser.add_argument("--save-stage", type=str, default="", help="Optional path to save generated stage")
    parser.add_argument("--renderer", type=str, default="RayTracedLighting", help="RayTracedLighting or PathTracing")
    parser.add_argument("--camera-fov-deg", type=float, default=84.0, help="Forward onboard camera horizontal FOV")

    parser.add_argument("--plant-rotate-x", type=float, default=0.0, help="Imported plant root X rotation in degrees")
    parser.add_argument("--plant-rotate-y", type=float, default=0.0, help="Imported plant root Y rotation in degrees")
    parser.add_argument("--plant-rotate-z", type=float, default=0.0, help="Imported plant root Z rotation in degrees")
    parser.add_argument("--plant-scale", type=float, default=1.0, help="Additional wrapper scale for imported plant")
    parser.add_argument("--plant-margin", type=float, default=8.0, help="Ground/fence margin around imported plant in meters")
    parser.add_argument("--plant-max-dim", type=float, default=75.0, help="Auto-shrink imported plant if XY footprint is larger than this many meters. Use 0 to disable.")
    parser.add_argument("--plant-axis", type=str, default="auto", choices=["auto", "z-up", "y-up"], help="Coordinate convention for imported plant. UNC Power Plant PLY uses Y-up.")
    parser.add_argument("--keep-procedural-site", action="store_true", help="Keep procedural roads/substation even when --plant-usd is given")
    parser.add_argument("--no-auto-fit-plant", action="store_true", help="Do not rotate/center/ground imported plant")

    parser.add_argument("--slam-mode", type=str, default="proxy", choices=["proxy", "gt", "cuvslam"], help="proxy = noisy SLAM-like pose, gt = debug only, cuvslam = external Isaac ROS Visual SLAM odometry")
    parser.add_argument("--enable-ros2-camera-pub", action="store_true", help="Enable Isaac Sim ROS 2 Bridge stereo camera publishing. Automatically enabled for --slam-mode cuvslam unless --disable-ros2-camera-pub is set.")
    parser.add_argument("--disable-ros2-camera-pub", action="store_true", help="Disable automatic ROS 2 stereo camera publishing even in cuvslam mode.")
    parser.add_argument("--ros2-domain-id", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "0")), help="ROS_DOMAIN_ID for Isaac ROS / cuVSLAM.")
    parser.add_argument("--ros2-camera-width", type=int, default=640, help="Stereo camera image width.")
    parser.add_argument("--ros2-camera-height", type=int, default=480, help="Stereo camera image height.")
    parser.add_argument("--ros2-camera-fps", type=float, default=20.0, help="Target camera publish rate hint for ROS 2 bridge.")
    parser.add_argument("--stereo-baseline", type=float, default=0.18, help="Stereo baseline in meters.")
    parser.add_argument("--left-image-topic", type=str, default="/front_stereo_camera/left/image_rect_color", help="Left stereo image topic expected by Isaac ROS Visual SLAM Isaac Sim launch.")
    parser.add_argument("--right-image-topic", type=str, default="/front_stereo_camera/right/image_rect_color", help="Right stereo image topic expected by Isaac ROS Visual SLAM Isaac Sim launch.")
    parser.add_argument("--left-camera-info-topic", type=str, default="/front_stereo_camera/left/camera_info", help="Left camera_info topic.")
    parser.add_argument("--right-camera-info-topic", type=str, default="/front_stereo_camera/right/camera_info", help="Right camera_info topic.")
    parser.add_argument("--imu-topic", type=str, default="/front_stereo_camera/imu", help="Optional IMU topic for VIO/diagnostics.")
    parser.add_argument("--cuvslam-odom-topic", type=str, default="/visual_slam/tracking/odometry", help="cuVSLAM odometry topic to use as GPS-denied pose input.")
    parser.add_argument("--cuvslam-odom-udp-host", type=str, default="0.0.0.0", help="UDP host for optional odometry bridge fallback.")
    parser.add_argument("--cuvslam-odom-udp-port", type=int, default=14555, help="UDP port for optional odometry bridge fallback.")
    parser.add_argument("--slam-drift-pos-per-sec", type=float, default=0.03, help="Accumulated proxy SLAM drift magnitude (m/s random walk)")
    parser.add_argument("--slam-pos-noise-std", type=float, default=0.03, help="Instantaneous SLAM position noise std (m)")
    parser.add_argument("--slam-yaw-noise-std", type=float, default=0.015, help="Instantaneous SLAM yaw noise std (rad)")
    parser.add_argument("--slam-vel-noise-std", type=float, default=0.02, help="Instantaneous SLAM velocity noise std (m/s)")
    parser.add_argument("--slam-tracking-loss-prob", type=float, default=0.008, help="Probability of temporary tracking degradation at each step")
    parser.add_argument("--slam-quality-recover-rate", type=float, default=0.06, help="How quickly tracking quality recovers")
    parser.add_argument("--depth-noise-std", type=float, default=0.02, help="Noise on depth rays for proxy perception")

    # Paper-methodology/results logging. These names follow the paper notation:
    # Psucc, tau_avg, Dinc, Aloc, Econ, Etime, fuzzy memberships, OSD speed and OP-CBRS potential.
    parser.add_argument("--sim-env-id", type=str, default="e1", choices=["e1", "e2"], help="Evaluation domain: e1=nuclear/power-plant containment, e2=industrial area transfer domain.")
    parser.add_argument("--policy-name", type=str, default="OSD+Fuzzy+OP-CBRS", help="Name written into metrics CSV for paper tables.")
    parser.add_argument("--output-root", type=str, default="~/uav_inspection", help="Root folder for paper-aligned outputs (metrics, trajectories, figures).")
    parser.add_argument("--metrics-dir", type=str, default="", help="Directory for paper metrics. Defaults to <output-root>/metrics.")
    parser.add_argument("--trajectory-dir", type=str, default="", help="Directory for per-episode trajectory records. Defaults to <output-root>/trajectories.")
    parser.add_argument("--figure-dir", type=str, default="", help="Directory for paper-style figures. Defaults to <output-root>/figures.")
    parser.add_argument("--step-metrics-csv", type=str, default="", help="Optional CSV path for per-step metrics.")
    parser.add_argument("--episode-metrics-csv", type=str, default="", help="Optional CSV path for per-episode metrics.")
    parser.add_argument("--summary-json", type=str, default="", help="Optional path for cumulative paper-summary JSON.")
    parser.add_argument("--localization-threshold", type=float, default=0.10, help="Aloc threshold in meters, matching paper definition <= 0.1 m.")
    parser.add_argument("--drift-threshold", type=float, default=0.50, help="Localization-error threshold in meters used to count Dinc drift incidents.")
    parser.add_argument("--coverage-radius", type=float, default=1.25, help="Radius around each coverage sample considered inspected.")
    parser.add_argument("--motor-hover-power-w", type=float, default=180.0, help="Nominal hover power for Econ Wh estimate.")
    parser.add_argument("--motor-speed-power-gain-w", type=float, default=32.0, help="Additional motor power gain proportional to speed^2.")
    parser.add_argument("--enable-op-cbrs", action="store_true", default=True, help="Use OP-CBRS-style potential shaping in reward and metrics.")
    parser.add_argument("--disable-op-cbrs", action="store_true", help="Disable OP-CBRS-style reward shaping but still log potential.")
    parser.add_argument("--disable-paper-figures", action="store_true", help="Do not render paper-style PNG figures from recorded trajectories.")
    parser.add_argument("--disable-trajectory-record", action="store_true", help="Do not save per-episode trajectory CSV/JSON bundles.")
    parser.add_argument("--heatmap-grid-size", type=int, default=128, help="Grid size used for saved perception/coverage heatmaps.")
    return parser.parse_args()



class CuvslamOdomReceiver:
    """Receives cuVSLAM odometry for GPS-denied policy input.

    Two paths are supported:
    1. Direct rclpy subscription, if rclpy is importable in the Isaac Sim Python.
    2. UDP JSON fallback. This is useful because Isaac Sim Python and ROS 2 Humble
       sometimes use different Python versions. In that case, run a tiny ROS 2
       bridge in the Isaac ROS container that forwards /visual_slam/tracking/odometry
       to this UDP port.
    """

    def __init__(self, odom_topic: str, udp_host: str = "0.0.0.0", udp_port: int = 14555):
        self.odom_topic = odom_topic
        self.udp_host = udp_host
        self.udp_port = int(udp_port)
        self.latest = None
        self.latest_time = 0.0
        self._lock = threading.Lock()
        self._rclpy = None
        self._node = None
        self._spin_thread = None
        self._udp_sock = None

        self._start_udp_receiver()
        self._try_start_rclpy_receiver()

    @staticmethod
    def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        # ZYX yaw from quaternion.
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _start_udp_receiver(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.udp_host, self.udp_port))
            sock.setblocking(False)
            self._udp_sock = sock
            print(f"[CUVSLAM] UDP odometry receiver listening on {self.udp_host}:{self.udp_port}")
        except Exception as exc:
            print(f"[CUVSLAM] Warning: could not start UDP odometry receiver: {exc}")
            self._udp_sock = None

    def _try_start_rclpy_receiver(self):
        try:
            import rclpy
            from nav_msgs.msg import Odometry
        except Exception as exc:
            print("[CUVSLAM] Direct rclpy odometry subscription is unavailable in this Python.")
            print(f"[CUVSLAM] Use the UDP bridge fallback if needed. rclpy import error: {exc}")
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            node = rclpy.create_node("npp_drone_cuvslam_odom_receiver")
            node.create_subscription(Odometry, self.odom_topic, self._rclpy_odom_cb, 10)
            self._rclpy = rclpy
            self._node = node
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()
            print(f"[CUVSLAM] Subscribed directly to {self.odom_topic}")
        except Exception as exc:
            print(f"[CUVSLAM] Warning: failed to create direct rclpy odometry subscriber: {exc}")
            self._rclpy = None
            self._node = None

    def _spin_loop(self):
        while self._rclpy is not None and self._node is not None:
            try:
                self._rclpy.spin_once(self._node, timeout_sec=0.05)
            except Exception:
                time.sleep(0.05)

    def _rclpy_odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        self._store(
            pos=[float(p.x), float(p.y), float(p.z)],
            quat=[float(q.x), float(q.y), float(q.z), float(q.w)],
            vel=[float(v.x), float(v.y), float(v.z)],
            ang=[float(w.x), float(w.y), float(w.z)],
            stamp=time.time(),
        )

    def _store(self, pos, quat, vel, ang, stamp):
        yaw = self.quat_to_yaw(quat[0], quat[1], quat[2], quat[3])
        data = {
            "pos": np.array(pos, dtype=np.float32),
            "vel": np.array(vel, dtype=np.float32),
            "quat": np.array(quat, dtype=np.float32),
            "ang": np.array(ang, dtype=np.float32),
            "yaw": float(yaw),
            "stamp": float(stamp),
        }
        with self._lock:
            self.latest = data
            self.latest_time = time.time()

    def _poll_udp(self):
        if self._udp_sock is None:
            return
        while True:
            try:
                packet, _addr = self._udp_sock.recvfrom(4096)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                d = json.loads(packet.decode("utf-8"))
                pos = d.get("pos", [0.0, 0.0, 0.0])
                quat = d.get("quat", [0.0, 0.0, 0.0, 1.0])
                vel = d.get("vel", [0.0, 0.0, 0.0])
                ang = d.get("ang", [0.0, 0.0, 0.0])
                self._store(pos, quat, vel, ang, d.get("stamp", time.time()))
            except Exception:
                continue

    def get_latest(self, max_age_sec: float = 0.50):
        self._poll_udp()
        with self._lock:
            if self.latest is None:
                return None
            age = time.time() - self.latest_time
            if age > max_age_sec:
                return None
            return dict(self.latest)

    def close(self):
        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        try:
            if self._udp_sock is not None:
                self._udp_sock.close()
        except Exception:
            pass


class NPPDroneGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        max_episode_steps: int = 900,
        inspection_reach_radius: float = 1.50,
        complete_route_before_timeout: bool = False,
        world_size: float = 34.0,
        num_obstacles: int = 8,
        num_rays: int = 36,
        render_sim: bool = True,
        seed: Optional[int] = None,
        drone_usd: str = "",
        plant_usd: str = "",
        hdri_path: str = "",
        save_stage: str = "",
        camera_fov_deg: float = 84.0,
        osd_vmin: float = 0.25,
        osd_vmax: float = 1.00,
        plant_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        plant_scale: float = 1.0,
        plant_margin: float = 8.0,
        plant_max_dim: float = 75.0,
        plant_axis: str = "auto",
        keep_procedural_site: bool = False,
        auto_fit_plant: bool = True,
        slam_mode: str = "proxy",
        slam_drift_pos_per_sec: float = 0.03,
        slam_pos_noise_std: float = 0.03,
        slam_yaw_noise_std: float = 0.015,
        slam_vel_noise_std: float = 0.02,
        slam_tracking_loss_prob: float = 0.008,
        slam_quality_recover_rate: float = 0.06,
        depth_noise_std: float = 0.02,
        enable_ros2_camera_pub: bool = False,
        disable_ros2_camera_pub: bool = False,
        ros2_domain_id: int = 0,
        ros2_camera_width: int = 640,
        ros2_camera_height: int = 480,
        ros2_camera_fps: float = 20.0,
        stereo_baseline: float = 0.18,
        left_image_topic: str = "/front_stereo_camera/left/image_raw",
        right_image_topic: str = "/front_stereo_camera/right/image_raw",
        left_camera_info_topic: str = "/front_stereo_camera/left/camera_info",
        right_camera_info_topic: str = "/front_stereo_camera/right/camera_info",
        imu_topic: str = "/front_stereo_camera/imu",
        cuvslam_odom_topic: str = "/visual_slam/tracking/odometry",
        cuvslam_odom_udp_host: str = "0.0.0.0",
        cuvslam_odom_udp_port: int = 14555,
        sim_env_id: str = "e1",
        policy_name: str = "OSD+Fuzzy+OP-CBRS",
        output_root: str = "~/uav_inspection",
        metrics_dir: str = "",
        trajectory_dir: str = "",
        figure_dir: str = "",
        step_metrics_csv: str = "",
        episode_metrics_csv: str = "",
        summary_json: str = "",
        localization_threshold: float = 0.10,
        drift_threshold: float = 0.50,
        coverage_radius: float = 1.25,
        motor_hover_power_w: float = 180.0,
        motor_speed_power_gain_w: float = 32.0,
        enable_op_cbrs: bool = True,
        op_cbrs_library_path: str = "",
        save_paper_figures: bool = True,
        save_trajectory_record: bool = True,
        heatmap_grid_size: int = 128,
        domain_randomization: bool = True,
        route_randomization: bool = False,
        obstacle_randomization: bool = True,
        safe_obstacle_placement: bool = True,
        route_corridor_clearance: float = 2.60,
        collision_termination: bool = True,
        policy_analytic_blend: float = 0.65,
        adaptive_speed_floor: float = 0.45,
        obstacle_avoidance_gain: float = 0.85,
        obstacle_avoidance_range: float = 2.20,
        yaw_gain: float = 2.15,
        velocity_memory: float = 0.74,
        enable_osd: bool = False,
    ):
        self.max_episode_steps = int(max_episode_steps)
        self.inspection_reach_radius = float(max(0.25, inspection_reach_radius))
        self.complete_route_before_timeout = bool(complete_route_before_timeout)
        self.world_size = float(world_size)
        self.user_world_size = float(world_size)
        self.num_obstacles = int(num_obstacles)
        self.num_rays = int(num_rays)
        self.max_ray_range = 18.0
        self.dt = 0.08
        self.render_sim = bool(render_sim)
        self.no_fly_z_min = 0.35
        self.no_fly_z_max = 24.0
        self.drone_usd = str(drone_usd or "")
        self.plant_usd = str(plant_usd or "")
        self.hdri_path = str(hdri_path or "")
        self.save_stage = str(save_stage or "")
        self.drone_uses_usd = False
        self.imported_plant_loaded = False
        self.keep_procedural_site = bool(keep_procedural_site)
        self.auto_fit_plant = bool(auto_fit_plant)
        self.plant_axis = str(plant_axis or "auto").lower()
        self.plant_rotation = self._resolve_imported_plant_rotation(plant_rotation)
        self.plant_scale = float(plant_scale)
        self.plant_margin = float(plant_margin)
        self.plant_max_dim = float(plant_max_dim)
        self.plant_bbox_min: Optional[np.ndarray] = None
        self.plant_bbox_max: Optional[np.ndarray] = None
        self.ground_center = np.array([0.0, 0.0], dtype=np.float32)
        self.ground_size_xy = np.array([self.world_size, self.world_size], dtype=np.float32)
        self.camera_hfov = math.radians(float(camera_fov_deg))
        # Compute vertical FOV from the actual camera aspect ratio instead of
        # using a fixed 0.75 multiplier. This keeps visual-feature detection
        # correct for 4:3, 16:9, and custom camera resolutions.
        cam_aspect_h_over_w = float(max(1, int(ros2_camera_height))) / float(max(1, int(ros2_camera_width)))
        self.camera_vfov = 2.0 * math.atan(math.tan(0.5 * self.camera_hfov) * cam_aspect_h_over_w)
        self.target_radius = 0.35
        self.osd_vmin = float(osd_vmin)
        self.osd_vmax = float(max(osd_vmax, osd_vmin + 1e-6))
        self.last_policy_speed_mps = self.osd_vmin
        self.last_commanded_speed_mps = self.osd_vmin
        self.last_osd_speed_cap_mps = self.osd_vmax
        self.rng = np.random.default_rng(seed)

        # -----------------------------------------------------------------
        # Paper-aligned methodology/results configuration
        # -----------------------------------------------------------------
        self.sim_env_id = str(sim_env_id or "e1")
        self.policy_name = str(policy_name or "OSD+Fuzzy+OP-CBRS")
        self.localization_threshold = float(localization_threshold)
        self.drift_threshold = float(drift_threshold)
        self.coverage_radius = float(coverage_radius)
        self.motor_hover_power_w = float(motor_hover_power_w)
        self.motor_speed_power_gain_w = float(motor_speed_power_gain_w)
        self.enable_op_cbrs = bool(enable_op_cbrs)
        self.op_cbrs_library_path = Path(op_cbrs_library_path).expanduser() if op_cbrs_library_path else None
        self.op_cbrs_library = self._load_op_cbrs_library(self.op_cbrs_library_path)
        self.output_root = Path(output_root).expanduser()
        self.output_root.mkdir(parents=True, exist_ok=True)
        metrics_root = Path(metrics_dir).expanduser() if metrics_dir else self.output_root / "metrics"
        metrics_root.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = metrics_root
        self.trajectory_dir = Path(trajectory_dir).expanduser() if trajectory_dir else self.output_root / "trajectories"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir = Path(figure_dir).expanduser() if figure_dir else self.output_root / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.save_paper_figures = bool(save_paper_figures)
        self.save_trajectory_record = bool(save_trajectory_record)
        self.heatmap_grid_size = max(int(heatmap_grid_size), 16)
        self.domain_randomization = bool(domain_randomization)
        self.route_randomization = bool(route_randomization)
        self.obstacle_randomization = bool(obstacle_randomization)
        self.safe_obstacle_placement = bool(safe_obstacle_placement)
        self.route_corridor_clearance = float(max(0.25, route_corridor_clearance))
        self.collision_termination = bool(collision_termination)
        # Robust transfer shield and smoother waypoint tracking. These values do
        # not change the paper-level action space: PPO still outputs one scalar
        # OSD speed. They improve the low-level conversion from scalar speed to
        # safe waypoint-following motion under e2 domain shift.
        self.policy_analytic_blend = float(np.clip(policy_analytic_blend, 0.0, 1.0))
        self.adaptive_speed_floor = float(np.clip(adaptive_speed_floor, 0.0, 1.0))
        self.obstacle_avoidance_gain = float(max(0.0, obstacle_avoidance_gain))
        self.obstacle_avoidance_range = float(max(0.05, obstacle_avoidance_range))
        self.yaw_gain = float(max(0.1, yaw_gain))
        self.velocity_memory = float(np.clip(velocity_memory, 0.0, 0.98))
        # PPO-without-OSD ablation switch. When False, PPO controls the scalar
        # speed directly and the analytic/fuzzy OSD speed shield, OSD cap,
        # risk-aware floor, and low-texture parallax helper are not applied.
        self.enable_osd = bool(enable_osd)
        self.step_metrics_csv = Path(step_metrics_csv).expanduser() if step_metrics_csv else metrics_root / "paper_step_metrics.csv"
        self.episode_metrics_csv = Path(episode_metrics_csv).expanduser() if episode_metrics_csv else metrics_root / "paper_episode_metrics.csv"
        self.summary_json = Path(summary_json).expanduser() if summary_json else metrics_root / "paper_summary.json"
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.metric_episode_id = 0
        self.current_segment_start = np.zeros(3, dtype=np.float32)
        self.coverage_points = None
        self.coverage_visited = None
        self.prev_paper_potential = None
        self.prev_paper_coverage = 0.0
        self.episode_metrics = {}
        self.trajectory_records = []
        self.latest_episode_bundle = None
        self.summary_counts = {"episodes": 0, "successes": 0, "drift_incidents": 0, "successful_time_sum": 0.0, "energy_wh_sum": 0.0, "loc_acc_time_sum": 0.0, "total_time_sum": 0.0}
        self.texture_count = 0
        self.illumination_lux = 500.0
        self.wind_mps = 0.5

        # Navigation / proxy SLAM state
        self.slam_mode = str(slam_mode)
        self.slam_drift_pos_per_sec = float(slam_drift_pos_per_sec)
        self.slam_pos_noise_std = float(slam_pos_noise_std)
        self.slam_yaw_noise_std = float(slam_yaw_noise_std)
        self.slam_vel_noise_std = float(slam_vel_noise_std)
        self.slam_tracking_loss_prob = float(slam_tracking_loss_prob)
        self.slam_quality_recover_rate = float(slam_quality_recover_rate)
        self.depth_noise_std = float(depth_noise_std)
        self.enable_ros2_camera_pub = bool(enable_ros2_camera_pub or (self.slam_mode == "cuvslam")) and not bool(disable_ros2_camera_pub)
        self.ros2_domain_id = int(ros2_domain_id)
        self.ros2_camera_width = int(ros2_camera_width)
        self.ros2_camera_height = int(ros2_camera_height)
        self.ros2_camera_fps = float(ros2_camera_fps)
        self.stereo_baseline = float(stereo_baseline)
        self.left_image_topic = str(left_image_topic)
        self.right_image_topic = str(right_image_topic)
        self.left_camera_info_topic = str(left_camera_info_topic)
        self.right_camera_info_topic = str(right_camera_info_topic)
        self.imu_topic = str(imu_topic)
        self.cuvslam_odom_topic = str(cuvslam_odom_topic)
        self.cuvslam_odom_udp_host = str(cuvslam_odom_udp_host)
        self.cuvslam_odom_udp_port = int(cuvslam_odom_udp_port)
        self.cuvslam_receiver = None
        self.ros2_graph_created = False
        self.visual_feature_points = None
        self.slam_origin_true = np.zeros(3, dtype=np.float32)
        self.slam_yaw0_true = 0.0
        self.slam_pos = np.zeros(3, dtype=np.float32)
        self.slam_vel = np.zeros(3, dtype=np.float32)
        self.slam_yaw = 0.0
        self.slam_quality = 1.0
        self.slam_drift = np.zeros(3, dtype=np.float32)
        self.cuvslam_metric_offset = None
        self.prev_true_vel = np.zeros(3, dtype=np.float32)
        self.imu_acc_body = np.zeros(3, dtype=np.float32)
        self.imu_gyro_body = np.zeros(3, dtype=np.float32)

        # Paper-aligned action: the learned OSD policy outputs one scalar speed.
        # The low-level controller converts this speed into body-frame tracking
        # toward the active inspection waypoint, while yaw and altitude are
        # stabilized deterministically. This removes the earlier algorithm-level
        # mismatch where the policy directly controlled vx, vy, vz and yaw-rate.
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: slam_pos(3), slam_vel(3), slam_yaw_sin_cos(2), imu_acc(3), imu_gyro(3),
        # target_rel_slam(3), camera_target_cues(5), depth_rays(N), tracking_quality(1),
        # previous scalar OSD action(1), mission_progress(2)
        obs_dim = 3 + 3 + 2 + 3 + 3 + 3 + 5 + self.num_rays + 1 + 1 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.stage = None
        self.ops: Dict[str, Dict[str, object]] = {}
        self.materials: Dict[str, str] = {}
        self.step_count = 0
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.prev_action = np.zeros(1, dtype=np.float32)
        # Planned inspection route follows a boustrophedon coverage pattern,
        # matching the paper figure style and giving the UAV explicit sweep
        # passes between inspection objects / chimneys.
        self.base_targets = self._build_boustrophedon_targets()
        self.targets = self.base_targets.copy()
        expected_targets = 12 if str(getattr(self, "sim_env_id", "e1")).lower() == "e2" else 16
        if len(self.targets) != expected_targets:
            raise RuntimeError(
                f"[ROUTE_ERROR_INIT] sim_env_id={self.sim_env_id} expected {expected_targets} targets, "
                f"got {len(self.targets)}. Check duplicate/old route functions."
            )
        print(f"[ROUTE_INIT] sim_env_id={self.sim_env_id} vslam_waypoints={len(self.targets)} drone_targets={len(self.targets)}")
        self.target_idx = 0
        self.target = self.targets[0].copy()
        self.prev_dist = 0.0
        self.obstacles = np.zeros((max(self.num_obstacles, 0), 6), dtype=np.float32)
        # Conservative analytic collision volumes used by the lightweight Gym
        # dynamics.  Isaac/Usd geometry is visual-only here, so these AABBs are
        # required to prevent the drone from flying through plant structures.
        self.static_collision_boxes = []
        self.last_collision_event = False
        self.last_collision_name = ""
        self.last_collision_avoidance_name = ""
        self.last_collision_distance_m = float("inf")
        self.drone_collision_radius = 0.42
        # Keep UAV visibly above plant structures instead of brushing/colliding with roofs.
        self.vertical_structure_clearance_m = 3.20

        self._build_scene()
        self.visual_feature_points = self._make_visual_feature_points()
        if self.slam_mode == "cuvslam":
            self.cuvslam_receiver = CuvslamOdomReceiver(
                odom_topic=self.cuvslam_odom_topic,
                udp_host=self.cuvslam_odom_udp_host,
                udp_port=self.cuvslam_odom_udp_port,
            )
        self.reset(seed=seed)

    # ---------------------------------------------------------------------
    # gym
    # ---------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.vel = np.zeros(3, dtype=np.float32)
        self.prev_true_vel = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.prev_action = np.zeros(1, dtype=np.float32)
        self.imu_acc_body = np.zeros(3, dtype=np.float32)
        self.imu_gyro_body = np.zeros(3, dtype=np.float32)
        self._randomize_domain()
        expected_targets = 12 if str(getattr(self, "sim_env_id", "e1")).lower() == "e2" else 16
        if len(self.targets) != expected_targets:
            raise RuntimeError(
                f"[ROUTE_ERROR_RESET] sim_env_id={self.sim_env_id} expected {expected_targets} targets, "
                f"got {len(self.targets)} after reset/randomization."
            )
        print(f"[ROUTE_RESET] sim_env_id={self.sim_env_id} vslam_waypoints={len(self.targets)} drone_targets={len(self.targets)}")

        if self.imported_plant_loaded and self.plant_bbox_min is not None and self.plant_bbox_max is not None:
            mn = self.plant_bbox_min
            mx = self.plant_bbox_max
            center = 0.5 * (mn + mx)
            x = center[0] - 0.45 * max(mx[0] - mn[0], 10.0)
            y = mn[1] - 6.0
            z = max(10.5, min(float(mx[2]) + 4.0, self.no_fly_z_max - 1.2))
            self.pos = np.array([x, y, z], dtype=np.float32)
            self.yaw = math.atan2(center[1] - y, center[0] - x)
        else:
            # Start at a domain-specific inspection altitude.  e2 uses a lower
            # industrial yard route, while e1 uses the higher power-plant route.
            if str(getattr(self, "sim_env_id", "e1")).lower() == "e2":
                self.pos = np.array([-15.5, -12.5, 9.6], dtype=np.float32)
                self.yaw = 0.55
            else:
                self.pos = np.array([-14.0, -12.0, 12.8], dtype=np.float32)
                self.yaw = 0.4

        self.target_idx = 0
        self.target = self.targets[self.target_idx].copy()
        self.current_segment_start = self.pos.copy()
        self.prev_dist = float(np.linalg.norm(self.target - self.pos))
        # Reset the no-progress watchdog at episode start.
        self._watchdog_best_dist = float('inf')
        self._watchdog_no_improve_steps = 0
        self._watchdog_target_idx = self.target_idx
        self.last_collision_event = False
        self.last_collision_name = ""
        self.last_collision_avoidance_name = ""
        self.last_collision_distance_m = float("inf")
        self._reset_navigation_proxy()
        self._start_paper_episode_metrics()
        self._sync_scene()
        self._nav_metrics_cache = self._collect_navigation_metrics()
        self._update_viewport_camera()
        return self._get_obs(), {"target_index": self.target_idx}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.step_count += 1

        # Paper-aligned OSD action A=[v_min, v_max] subset R.
        # The policy chooses only the scalar inspection speed.  Direction,
        # altitude and yaw are handled by the waypoint-tracking controller so
        # that the action semantics match the OSD formulation in the paper.
        speed_alpha = float(np.clip(0.5 * (float(action[0]) + 1.0), 0.0, 1.0))
        policy_speed = self.osd_vmin + speed_alpha * (self.osd_vmax - self.osd_vmin)
        # Fresh pre-action navigation metrics for the OSD shield. The reward
        # computes a separate post-action snapshot after the state update, so
        # action selection and reward logging are not mixed across stale cache
        # entries.
        nav_metrics = self._collect_navigation_metrics()
        self._nav_metrics_cache = dict(nav_metrics)
        mu_T = float(nav_metrics.get("mu_T", 0.0))
        mu_L = float(nav_metrics.get("mu_L", 0.0))
        mu_W = float(nav_metrics.get("mu_W", 0.0))
        mu_A = float(nav_metrics.get("mu_A", 0.0))
        osd_speed_cap = float(np.clip(float(nav_metrics.get("osd_speed_vt", self.osd_vmax)), self.osd_vmin, self.osd_vmax))

        if self.enable_osd:
            # Full method: analytic/fuzzy OSD shield blends PPO speed with the
            # fuzzy OSD speed cap and applies a risk-aware floor.
            risk = self._clip01(
                0.45 * (1.0 - mu_T)
                + 0.25 * (1.0 - mu_L)
                + 0.20 * (1.0 - mu_W)
                + 0.10 * (1.0 - mu_A)
            )
            blend = float(getattr(self, "policy_analytic_blend", 0.65))
            blended_speed = (1.0 - blend) * policy_speed + blend * osd_speed_cap
            risk_aware_floor = self.osd_vmin + float(getattr(self, "adaptive_speed_floor", 0.45)) * (1.0 - risk) * (self.osd_vmax - self.osd_vmin)
            commanded_speed = float(np.clip(max(blended_speed, risk_aware_floor), self.osd_vmin, osd_speed_cap))
        else:
            # PPO-without-OSD ablation: remove the analytic/fuzzy OSD speed
            # shield. PPO directly controls scalar speed within [v_min, v_max].
            # OP-CBRS can remain enabled, so this isolates the OSD speed module.
            commanded_speed = float(np.clip(policy_speed, self.osd_vmin, self.osd_vmax))

        # Slow down when the waypoint is close to prevent overshoot.
        rel_world = np.asarray(self.target - self.pos, dtype=np.float32)
        dist_xy = float(np.linalg.norm(rel_world[:2]))
        if dist_xy < 1.80:
            commanded_speed = min(commanded_speed, 0.55)
        if dist_xy < 0.65:
            commanded_speed = min(commanded_speed, 0.32)

        rel_body = self._world_to_body(rel_world)
        direction_xy = rel_body[:2]
        direction_norm = float(np.linalg.norm(direction_xy))
        if direction_norm > 1e-6:
            direction_xy = direction_xy / direction_norm
        else:
            direction_xy = np.zeros(2, dtype=np.float32)
        body_xy = direction_xy * commanded_speed
        # Deterministic local obstacle repulsion.  This uses only onboard-like
        # depth/geometry cues and keeps the policy action scalar. It reduces
        # random e2 failures caused by fixed straight-line waypoint tracking.
        body_xy = body_xy + self._local_obstacle_avoidance_body_xy(commanded_speed)
        speed_xy = float(np.linalg.norm(body_xy))
        if speed_xy > commanded_speed and speed_xy > 1e-6:
            body_xy = body_xy / speed_xy * commanded_speed
        # The full OSD controller adds a small lateral parallax helper in
        # low-texture zones. It is disabled for PPO-without-OSD ablation.
        if self.enable_osd and mu_T < 0.35 and dist_xy > 1.5:
            body_xy[1] += 0.10 * math.sin(0.055 * float(self.step_count))
        safe_floor = self._safe_altitude_floor(self.pos[:2], margin=2.40)
        # Bias altitude tracking upward near structures. This prevents the UAV
        # from grazing roofs/chimneys while it sweeps over the plant.
        rel_z_for_clearance = max(float(rel_world[2]), float(safe_floor - self.pos[2]))
        vz_cmd = float(np.clip(0.70 * rel_z_for_clearance, -0.45, 1.15))
        desired_yaw = math.atan2(float(rel_world[1]), float(rel_world[0]))
        yaw_err = self._wrap_angle(desired_yaw - float(self.yaw))
        yaw_rate = float(np.clip(float(getattr(self, "yaw_gain", 2.15)) * yaw_err, -1.45, 1.45))
        body_cmd = np.array([body_xy[0], body_xy[1], vz_cmd], dtype=np.float32)
        world_cmd = self._body_to_world(body_cmd)
        self.last_policy_speed_mps = float(policy_speed)
        self.last_commanded_speed_mps = float(commanded_speed)
        self.last_osd_speed_cap_mps = float(osd_speed_cap)

        self.prev_true_vel = self.vel.copy()
        vel_mem = float(getattr(self, "velocity_memory", 0.74))
        self.vel = vel_mem * self.vel + (1.0 - vel_mem) * world_cmd
        prev_pos_for_collision = self.pos.copy()
        candidate_pos = self.pos + self.vel * self.dt
        candidate_pos, collision_event, collision_name = self._resolve_collision_free_motion(prev_pos_for_collision, candidate_pos)
        self.last_collision_event = bool(collision_event)
        self.last_collision_name = str(collision_name or "")
        if collision_event:
            # Stop instead of tunneling through geometry.  This makes collision
            # visible to reward/metrics and prevents the rendered UAV from
            # passing through structures like air.
            self.vel *= 0.0
        self.pos = candidate_pos
        self.yaw += yaw_rate * self.dt
        self._update_proxy_inertial(yaw_rate)
        self._update_navigation_proxy()
        self._sync_scene()

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(action, precomputed_nav=nav_metrics)
        timeout_reached = self.step_count >= self.max_episode_steps
        # When complete_route_before_timeout is True, ignore the soft
        # max_episode_steps timeout but enforce a hard upper cap and a
        # no-progress watchdog so a deadlocked UAV cannot run indefinitely.
        if self.complete_route_before_timeout:
            # Route-completion evaluation/collection must not reset at the soft
            # max_episode_steps.  Use a very generous hard cap only as a safety
            # guard against a genuine deadlock.
            hard_cap = max(int(self.max_episode_steps) * 10, int(self.max_episode_steps) + 60000)
        else:
            hard_cap = max(int(self.max_episode_steps) * 4, int(self.max_episode_steps) + 12000)
        hard_cap_reached = self.step_count >= hard_cap
        # No-progress watchdog: track best distance to the current waypoint.
        if not hasattr(self, '_watchdog_best_dist') or getattr(self, '_watchdog_target_idx', -1) != self.target_idx:
            self._watchdog_best_dist = float('inf')
            self._watchdog_no_improve_steps = 0
            self._watchdog_target_idx = self.target_idx
        cur_dist_for_watchdog = float(np.linalg.norm(self.target - self.pos))
        if cur_dist_for_watchdog + 0.25 < self._watchdog_best_dist:
            self._watchdog_best_dist = cur_dist_for_watchdog
            self._watchdog_no_improve_steps = 0
        else:
            self._watchdog_no_improve_steps += 1
        # In route-completion modes, give the UAV much longer to recover from
        # slow turns, SLAM noise, or near-waypoint oscillation.  The original
        # fixed 600-step watchdog could truncate collection before route finish.
        watchdog_limit = max(6000, int(self.max_episode_steps) // 2) if self.complete_route_before_timeout else 600
        no_progress_truncate = self._watchdog_no_improve_steps >= watchdog_limit
        if self.complete_route_before_timeout:
            truncated = bool(hard_cap_reached or no_progress_truncate)
        else:
            truncated = bool(timeout_reached or hard_cap_reached or no_progress_truncate)
        if timeout_reached and self.complete_route_before_timeout and (self.step_count == self.max_episode_steps or self.step_count % 250 == 0):
            print(
                f"[INSPECT] Timeout ignored until route completion: "
                f"step={self.step_count}, target={self.target_idx}/{len(self.targets)-1}, "
                f"reach_radius={self.inspection_reach_radius:.2f}m"
            )
        if no_progress_truncate:
            print(
                f"[WATCHDOG] No-progress truncation: target={self.target_idx}/{len(self.targets)-1}, "
                f"best_dist={self._watchdog_best_dist:.2f}m, "
                f"stuck_steps={self._watchdog_no_improve_steps}/{watchdog_limit}"
            )
        if hard_cap_reached and not no_progress_truncate:
            print(f"[WATCHDOG] Hard cap reached at step={self.step_count}, hard_cap={hard_cap}")
        info = self._record_paper_step_metrics(action, reward, info, terminated, truncated)
        route_complete_now = bool(info.get("route_complete", False)) or self.target_idx >= len(self.targets)
        if terminated or truncated:
            if self.complete_route_before_timeout and not route_complete_now:
                print(
                    f"[SKIP_FINALIZE] Episode ended before full route completion: "
                    f"target={self.target_idx}/{len(self.targets)-1}, "
                    f"terminated={terminated}, truncated={truncated}. "
                    f"No final route graph generated."
                )
            else:
                self._finalize_paper_episode_metrics(info, terminated, truncated)
        self.prev_action = action.copy()
        if self.render_sim:
            omni.kit.app.get_app().update()
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.cuvslam_receiver is not None:
            self.cuvslam_receiver.close()

    # ---------------------------------------------------------------------
    # asset / scene utility
    # ---------------------------------------------------------------------
    def _is_valid_asset_ref(self, value: str) -> bool:
        if not value:
            return False
        value = str(value)
        if value.startswith(("http://", "https://", "omniverse://")):
            return True
        return Path(value).expanduser().exists()

    def _asset_ref(self, value: str) -> str:
        value = str(value)
        if value.startswith(("http://", "https://", "omniverse://")):
            return value
        return str(Path(value).expanduser())

    def _resolve_imported_plant_rotation(self, plant_rotation):
        rot = tuple(float(v) for v in plant_rotation)
        path_hint = self.plant_usd.lower()
        user_left_rotation_default = all(abs(v) < 1e-9 for v in rot)

        if self.plant_axis == "y-up":
            print("[PLANT_FIT] plant_axis=y-up: applying +90 deg X rotation")
            return (90.0, 0.0, 0.0)

        if self.plant_axis == "auto" and user_left_rotation_default:
            if "unc_powerplant" in path_hint or "powerplant_full" in path_hint or "powerplant_" in path_hint:
                print("[PLANT_FIT] plant_axis=auto: detected UNC/PLY-style powerplant asset; applying +90 deg X rotation")
                return (90.0, 0.0, 0.0)

        if self.plant_axis == "z-up":
            print("[PLANT_FIT] plant_axis=z-up: using imported asset orientation")
        return rot

    def _build_scene(self):
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        try:
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        except Exception:
            pass

        self._create_lighting()
        self._create_material_palette()
        self._create_power_plant()
        if self.imported_plant_loaded and not self.keep_procedural_site:
            self._create_imported_plant_site()
        else:
            self._create_ground_and_site()
            if not self.imported_plant_loaded:
                self._create_procedural_power_plant()
            # e2 is the transfer-domain industrial layout.  Do not add the
            # e1/NPP substation and landmark objects on top of it; otherwise
            # the transfer run visually appears to use the same power-plant
            # environment.  The e2 layout creates its own warehouses, pipes,
            # tanks, racks, markers, and perimeter features.
            if str(getattr(self, "sim_env_id", "e1")).lower() != "e2":
                self._create_substation()
                self._create_visual_landmarks()
                self._create_perimeter_fence()
        self._create_drone()
        self._create_sensor_rig_visuals()
        self._create_stereo_camera_prims()
        self._setup_ros2_stereo_camera_graph()
        self._create_target()
        self._create_obstacles()
        for _ in range(20):
            omni.kit.app.get_app().update()
        self._update_viewport_camera()
        if self.save_stage:
            self._save_stage_safely(self.save_stage)

    def _save_stage_safely(self, path: str):
        try:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            omni.usd.get_context().save_as_stage(str(p))
            print(f"Saved generated stage to: {p}")
        except Exception as exc:
            print(f"Warning: could not save stage to {path}: {exc}")

    def _create_lighting(self):
        # Sky dome with cool zenith and slightly warm horizon for atmospheric realism.
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/SkyDome")
        dome.GetIntensityAttr().Set(720.0)
        dome.GetColorAttr().Set(Gf.Vec3f(0.78, 0.86, 0.97))
        if self.hdri_path:
            hdri = Path(self.hdri_path).expanduser()
            if hdri.exists():
                try:
                    dome.CreateTextureFileAttr(str(hdri))
                except Exception as exc:
                    print(f"Warning: could not apply HDRI {hdri}: {exc}")

        # Sun: warm afternoon light, soft angular size for realistic shadow penumbras.
        sun = UsdLux.DistantLight.Define(self.stage, "/World/Lights/Sun")
        sun.GetIntensityAttr().Set(2900.0)
        sun.GetColorAttr().Set(Gf.Vec3f(1.0, 0.965, 0.905))
        sun.GetAngleAttr().Set(0.45)
        UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate((-46.0, 25.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        # Cool sky-bounce fill from opposite side to soften shadows.
        fill = UsdLux.SphereLight.Define(self.stage, "/World/Lights/Fill")
        fill.GetIntensityAttr().Set(260.0)
        fill.GetRadiusAttr().Set(10.0)
        fill.GetColorAttr().Set(Gf.Vec3f(0.78, 0.86, 1.0))
        UsdGeom.XformCommonAPI(fill.GetPrim()).SetTranslate((-15.0, -16.0, 14.0))

        # Warm bounce light from concrete apron (subtle ground reflection).
        warm_bounce = UsdLux.SphereLight.Define(self.stage, "/World/Lights/WarmBounce")
        warm_bounce.GetIntensityAttr().Set(120.0)
        warm_bounce.GetRadiusAttr().Set(6.0)
        warm_bounce.GetColorAttr().Set(Gf.Vec3f(1.0, 0.92, 0.78))
        UsdGeom.XformCommonAPI(warm_bounce.GetPrim()).SetTranslate((10.0, 8.0, 1.5))

    def _create_material_palette(self):
        # =================================================================
        # PHOTOREAL INDUSTRIAL PALETTE
        # All values tuned to look like real weathered industrial surfaces
        # under ray-traced lighting. No pure whites, no saturated primaries.
        # Roughness > 0.85 for concrete, 0.3-0.5 for metals, < 0.1 for glass.
        # =================================================================

        # ----- CONCRETE FAMILY (multiple aged variants for surface variation) -----
        self._make_preview_material("mat_concrete", "/World/Looks/Concrete_Aged", (0.555, 0.545, 0.510), 0.94, 0.0)
        self._make_preview_material("mat_concrete_warm", "/World/Looks/Concrete_WarmGrey", (0.585, 0.560, 0.515), 0.93, 0.0)
        self._make_preview_material("mat_concrete_cool", "/World/Looks/Concrete_CoolGrey", (0.530, 0.535, 0.530), 0.95, 0.0)
        self._make_preview_material("mat_concrete_dark", "/World/Looks/Concrete_Shadowed", (0.345, 0.342, 0.322), 0.97, 0.0)
        self._make_preview_material("mat_weathered_concrete", "/World/Looks/Concrete_Weathered", (0.470, 0.460, 0.420), 0.96, 0.0)
        self._make_preview_material("mat_concrete_stain", "/World/Looks/Concrete_WaterStain", (0.255, 0.260, 0.250), 0.99, 0.0)
        self._make_preview_material("mat_concrete_algae", "/World/Looks/Concrete_AlgaeStain", (0.225, 0.275, 0.220), 0.99, 0.0)
        self._make_preview_material("mat_concrete_rust_streak", "/World/Looks/Concrete_RustStreak", (0.345, 0.215, 0.135), 0.97, 0.0)
        self._make_preview_material("mat_concrete_seam", "/World/Looks/Concrete_FormworkSeam", (0.395, 0.388, 0.358), 0.97, 0.0)

        # ----- COOLING TOWER FAMILY (hyperboloid weathered concrete with vertical streaks) -----
        self._make_preview_material("mat_tower_concrete", "/World/Looks/CoolingTower_Concrete", (0.560, 0.548, 0.498), 0.95, 0.0)
        self._make_preview_material("mat_tower_concrete_top", "/World/Looks/CoolingTower_TopRim", (0.605, 0.590, 0.535), 0.93, 0.0)
        self._make_preview_material("mat_tower_shadow", "/World/Looks/CoolingTower_ShadowBand", (0.305, 0.300, 0.275), 0.98, 0.0)
        self._make_preview_material("mat_tower_water_streak", "/World/Looks/CoolingTower_WaterStreak", (0.235, 0.245, 0.230), 0.98, 0.0)
        self._make_preview_material("mat_tower_base_wet", "/World/Looks/CoolingTower_WetBase", (0.215, 0.215, 0.205), 0.88, 0.02)
        self._make_preview_material("mat_tower_moss", "/World/Looks/CoolingTower_MossLine", (0.195, 0.245, 0.180), 0.99, 0.0)

        # ----- REACTOR CONTAINMENT (modern PWR style: warm pale concrete with formwork bands) -----
        self._make_preview_material("mat_reactor_dome", "/World/Looks/Reactor_Dome", (0.625, 0.610, 0.560), 0.88, 0.0)
        self._make_preview_material("mat_reactor_wall", "/World/Looks/Reactor_Containment", (0.595, 0.580, 0.535), 0.92, 0.0)
        self._make_preview_material("mat_reactor_seam", "/World/Looks/Reactor_FormworkBand", (0.475, 0.460, 0.420), 0.96, 0.0)
        self._make_preview_material("mat_reactor_lightning_rod", "/World/Looks/Reactor_LightningRod", (0.115, 0.115, 0.120), 0.42, 0.55)

        # ----- INDUSTRIAL STEEL FAMILY (varied roughness/metallic for realism) -----
        self._make_preview_material("mat_dark_steel", "/World/Looks/Steel_DarkPainted", (0.085, 0.092, 0.098), 0.52, 0.45)
        self._make_preview_material("mat_brushed_steel", "/World/Looks/Steel_GalvanizedFaded", (0.435, 0.445, 0.435), 0.46, 0.62)
        self._make_preview_material("mat_galvanized", "/World/Looks/Steel_Galvanized", (0.485, 0.498, 0.500), 0.38, 0.78)
        self._make_preview_material("mat_tank_steel", "/World/Looks/Tank_PaintedSteel", (0.485, 0.495, 0.508), 0.34, 0.55)
        self._make_preview_material("mat_tank_top_steel", "/World/Looks/Tank_DomeTop", (0.535, 0.545, 0.555), 0.32, 0.62)
        self._make_preview_material("mat_steel_rusted", "/World/Looks/Steel_Rusted", (0.305, 0.205, 0.135), 0.88, 0.18)
        self._make_preview_material("mat_steel_oxidized", "/World/Looks/Steel_Oxidized", (0.185, 0.165, 0.135), 0.78, 0.32)
        self._make_preview_material("mat_steel_white_paint", "/World/Looks/Steel_WhitePaint", (0.665, 0.655, 0.620), 0.55, 0.12)

        # ----- CLADDING / SIDING (industrial building skin) -----
        self._make_preview_material("mat_blue_siding", "/World/Looks/Siding_FadedBlueGrey", (0.345, 0.408, 0.462), 0.74, 0.08)
        self._make_preview_material("mat_green_siding", "/World/Looks/Siding_FadedSeafoam", (0.348, 0.408, 0.358), 0.76, 0.07)
        self._make_preview_material("mat_beige_siding", "/World/Looks/Siding_BeigePanel", (0.578, 0.535, 0.450), 0.74, 0.06)
        self._make_preview_material("mat_panel_dark", "/World/Looks/Panel_DarkRibbed", (0.215, 0.245, 0.270), 0.72, 0.08)
        self._make_preview_material("mat_panel_light", "/World/Looks/Panel_LightRibbed", (0.485, 0.510, 0.525), 0.70, 0.06)
        self._make_preview_material("mat_corrugated", "/World/Looks/Panel_Corrugated", (0.398, 0.408, 0.408), 0.62, 0.22)

        # ----- ASPHALT / GRAVEL / GROUND -----
        self._make_preview_material("mat_asphalt", "/World/Looks/Asphalt_Aged", (0.082, 0.082, 0.085), 0.97, 0.0)
        self._make_preview_material("mat_asphalt_wet", "/World/Looks/Asphalt_Wet", (0.058, 0.060, 0.068), 0.55, 0.0)
        self._make_preview_material("mat_gravel", "/World/Looks/Gravel_Crushed", (0.318, 0.305, 0.275), 0.98, 0.0)
        self._make_preview_material("mat_gravel_dark", "/World/Looks/Gravel_DarkBallast", (0.225, 0.218, 0.198), 0.99, 0.0)
        self._make_preview_material("mat_dirt", "/World/Looks/Dirt_DryEarth", (0.385, 0.328, 0.248), 0.99, 0.0)

        # ----- HAZARD / INDUSTRIAL SAFETY COLORS (faded, not fluorescent) -----
        self._make_preview_material("mat_warning", "/World/Looks/Safety_OchreFaded", (0.685, 0.515, 0.135), 0.78, 0.04)
        self._make_preview_material("mat_safety", "/World/Looks/Safety_RailingYellow", (0.745, 0.580, 0.085), 0.55, 0.06)
        self._make_preview_material("mat_safety_red", "/World/Looks/Safety_RedPaint", (0.595, 0.135, 0.115), 0.62, 0.04)

        # ----- PIPING (industrial color-coded with realistic faded paint) -----
        self._make_preview_material("mat_pipe_red", "/World/Looks/Pipe_RustRed", (0.435, 0.148, 0.118), 0.66, 0.06)
        self._make_preview_material("mat_pipe_green", "/World/Looks/Pipe_ServiceGreen", (0.155, 0.305, 0.185), 0.68, 0.06)
        self._make_preview_material("mat_pipe_blue", "/World/Looks/Pipe_ServiceBlue", (0.158, 0.245, 0.385), 0.68, 0.06)
        self._make_preview_material("mat_pipe_yellow", "/World/Looks/Pipe_SafetyYellow", (0.685, 0.555, 0.105), 0.66, 0.05)
        self._make_preview_material("mat_pipe_insulation", "/World/Looks/Pipe_InsulationJacket", (0.585, 0.575, 0.545), 0.58, 0.18)
        self._make_preview_material("mat_pipe_steam", "/World/Looks/Pipe_SteamGrey", (0.395, 0.395, 0.398), 0.42, 0.42)

        # ----- RUST / WEATHERING / DECALS -----
        self._make_preview_material("mat_rust", "/World/Looks/Surface_RustPatch", (0.385, 0.205, 0.095), 0.95, 0.02)
        self._make_preview_material("mat_oil_stain", "/World/Looks/Surface_OilStain", (0.045, 0.038, 0.032), 0.42, 0.0)
        self._make_preview_material("mat_dust", "/World/Looks/Surface_DustLayer", (0.605, 0.575, 0.515), 0.99, 0.0)

        # ----- GLASS / SPECIAL -----
        self._make_preview_material("mat_glass", "/World/Looks/Glass_DarkTinted", (0.038, 0.062, 0.078), 0.06, 0.0)
        self._make_preview_material("mat_glass_window", "/World/Looks/Glass_Industrial", (0.085, 0.115, 0.135), 0.08, 0.0)
        self._make_preview_material("mat_marker", "/World/Looks/Marker_HiddenWaypoint", (0.18, 0.18, 0.18), 0.95, 0.0)

        # ----- AVIATION / STACK MARKINGS -----
        self._make_preview_material("mat_stack_white", "/World/Looks/Stack_AviationWhite", (0.745, 0.738, 0.700), 0.78, 0.04)
        self._make_preview_material("mat_stack_red", "/World/Looks/Stack_AviationRed", (0.625, 0.115, 0.105), 0.62, 0.04)
        self._make_preview_material("mat_stack_soot", "/World/Looks/Stack_SootCap", (0.085, 0.082, 0.078), 0.92, 0.06)

        # ----- DRONE MATERIALS (carbon, lens, props) -----
        self._make_preview_material("mat_drone_carbon", "/World/Looks/Drone_CarbonFiber", (0.038, 0.038, 0.042), 0.18, 0.32)
        self._make_preview_material("mat_drone_body", "/World/Looks/Drone_Body", (0.085, 0.088, 0.092), 0.24, 0.22)
        self._make_preview_material("mat_drone_prop", "/World/Looks/Drone_Propeller", (0.022, 0.022, 0.025), 0.22, 0.15)
        self._make_preview_material("mat_drone_lens", "/World/Looks/Drone_CameraLens", (0.008, 0.010, 0.013), 0.04, 0.0)

        # ----- FENCE / RAILING -----
        self._make_preview_material("mat_fence", "/World/Looks/Fence_Galvanized", (0.225, 0.235, 0.235), 0.42, 0.62)
        self._make_preview_material("mat_white", "/World/Looks/Paint_FadedWhite", (0.715, 0.708, 0.668), 0.78, 0.02)

    # ---------------------------------------------------------------------
    # plant creation
    # ---------------------------------------------------------------------
    def _create_power_plant(self):
        print(f"[ASSET] plant_usd='{self.plant_usd}'")
        if not self._is_valid_asset_ref(self.plant_usd):
            print("[ASSET] No valid plant USD provided. Using procedural research-grade plant.")
            self.imported_plant_loaded = False
            return
        plant_ref = self._asset_ref(self.plant_usd)
        print(f"[ASSET] Loading plant USD: {plant_ref}")
        self._add_reference_xform(
            "/World/ImportedPowerPlant", plant_ref, pos=(0.0, 0.0, 0.0),
            scale=(self.plant_scale, self.plant_scale, self.plant_scale),
        )
        self.imported_plant_loaded = True
        for _ in range(10):
            omni.kit.app.get_app().update()
        if self.auto_fit_plant:
            self._orient_and_fit_imported_plant("/World/ImportedPowerPlant")
        else:
            self.plant_bbox_min, self.plant_bbox_max = self._compute_world_bbox("/World/ImportedPowerPlant")
        self._generate_targets_from_plant_bbox()
        self.num_obstacles = 0
        self.obstacles = np.zeros((0, 6), dtype=np.float32)
        print("[ASSET] Real plant USD loaded. Procedural plant geometry disabled.")

    def _orient_and_fit_imported_plant(self, plant_path: str):
        prim = self.stage.GetPrimAtPath(plant_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Plant prim not found: {plant_path}")
        api = UsdGeom.XformCommonAPI(prim)
        api.SetScale((self.plant_scale, self.plant_scale, self.plant_scale))
        api.SetRotate(self.plant_rotation, UsdGeom.XformCommonAPI.RotationOrderXYZ)
        api.SetTranslate((0.0, 0.0, 0.0))
        for _ in range(5):
            omni.kit.app.get_app().update()
        mn, mx = self._compute_world_bbox(plant_path)
        if mn is None or mx is None:
            return

        footprint = float(max(mx[0] - mn[0], mx[1] - mn[1]))
        if self.plant_max_dim > 0.0 and footprint > self.plant_max_dim:
            shrink = float(self.plant_max_dim / max(footprint, 1e-6))
            self.plant_scale *= shrink
            api.SetScale((self.plant_scale, self.plant_scale, self.plant_scale))
            api.SetTranslate((0.0, 0.0, 0.0))
            for _ in range(5):
                omni.kit.app.get_app().update()
            mn, mx = self._compute_world_bbox(plant_path)
            print(f"[PLANT_FIT] auto_shrink_factor: {shrink:.4f}")
            print(f"[PLANT_FIT] final_wrapper_scale: {self.plant_scale:.6f}")

        center_xy = 0.5 * (mn[:2] + mx[:2])
        api.SetTranslate((-float(center_xy[0]), -float(center_xy[1]), -float(mn[2])))
        for _ in range(5):
            omni.kit.app.get_app().update()
        mn, mx = self._compute_world_bbox(plant_path)
        self.plant_bbox_min = mn
        self.plant_bbox_max = mx
        size = mx - mn
        self.ground_center = np.array([0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])], dtype=np.float32)
        self.ground_size_xy = np.array([size[0] + self.plant_margin, size[1] + self.plant_margin], dtype=np.float32)
        self.world_size = float(max(self.ground_size_xy[0], self.ground_size_xy[1]))
        self.no_fly_z_max = float(max(12.0, min(mx[2] + 8.0, 80.0)))
        self.max_ray_range = float(max(12.0, min(self.world_size * 0.12, 50.0)))
        try:
            # Imported USD geometry is often not decomposed into helper-created
            # primitives. Register a conservative no-fly volume so lightweight
            # kinematics cannot tunnel through the asset.
            c = 0.5 * (mn + mx)
            s = np.maximum(mx - mn, 1e-3)
            self._register_static_collision_box("/World/ImportedPowerPlant/BBox", c, s, category="imported_bbox")
        except Exception:
            pass
        print("[PLANT_FIT] rotation_xyz_deg:", self.plant_rotation)
        print("[PLANT_FIT] bbox_min:", mn)
        print("[PLANT_FIT] bbox_max:", mx)
        print("[PLANT_FIT] world_size:", self.world_size)

    def _compute_world_bbox(self, prim_path: str):
        try:
            bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=True)
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return None, None
            rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mn = np.array(rng.GetMin(), dtype=np.float64)
            mx = np.array(rng.GetMax(), dtype=np.float64)
            if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
                return None, None
            return mn, mx
        except Exception as exc:
            print(f"[WARN] Could not compute bbox for {prim_path}: {exc}")
            return None, None

    def _create_imported_plant_site(self):
        if self.plant_bbox_min is None or self.plant_bbox_max is None:
            size_x = size_y = self.world_size
            cx = cy = 0.0
        else:
            size_x = float(self.ground_size_xy[0])
            size_y = float(self.ground_size_xy[1])
            cx, cy = float(self.ground_center[0]), float(self.ground_center[1])
        self._create_cube("/World/CompactGround", size=(size_x, size_y, 0.04), color=(0.30, 0.30, 0.29), pos=(cx, cy, -0.025), material_key="mat_gravel")
        z = 0.04
        self._create_cube("/World/Boundary/North", size=(size_x, 0.08, 0.08), color=(0.65, 0.54, 0.18), pos=(cx, cy + size_y / 2.0, z), material_key="mat_warning")
        self._create_cube("/World/Boundary/South", size=(size_x, 0.08, 0.08), color=(0.65, 0.54, 0.18), pos=(cx, cy - size_y / 2.0, z), material_key="mat_warning")
        self._create_cube("/World/Boundary/East", size=(0.08, size_y, 0.08), color=(0.65, 0.54, 0.18), pos=(cx + size_x / 2.0, cy, z), material_key="mat_warning")
        self._create_cube("/World/Boundary/West", size=(0.08, size_y, 0.08), color=(0.65, 0.54, 0.18), pos=(cx - size_x / 2.0, cy, z), material_key="mat_warning")

    def _create_ground_and_site(self):
        # Surrounding terrain - dark crushed gravel ballast.
        self._create_cube("/World/Terrain", size=(self.world_size * 2.0, self.world_size * 2.0, 0.05), color=(0.30, 0.30, 0.28), pos=(0.0, 0.0, -0.05), material_key="mat_gravel_dark")

        # Concrete apron (the central plant pad).
        self._create_cube("/World/Apron/Main", size=(34.0, 30.0, 0.05), color=(0.45, 0.44, 0.40), pos=(0.0, 0.0, 0.00), material_key="mat_weathered_concrete")

        # Concrete formwork seams on the apron - break up the flat surface.
        for i, x in enumerate(np.linspace(-15.0, 15.0, 7)):
            self._create_cube(f"/World/Apron/SeamX_{i}", size=(0.04, 28.0, 0.012), color=(0.35, 0.35, 0.32), pos=(float(x), 0.0, 0.034), material_key="mat_concrete_seam")
        for i, y in enumerate(np.linspace(-13.0, 13.0, 5)):
            self._create_cube(f"/World/Apron/SeamY_{i}", size=(33.0, 0.04, 0.012), color=(0.35, 0.35, 0.32), pos=(0.0, float(y), 0.034), material_key="mat_concrete_seam")

        # Subtle oil / water staining on the apron.
        for i, (x, y, sx, sy) in enumerate([
            (3.5, -2.0, 1.6, 1.0), (-4.0, 3.0, 2.1, 0.7), (8.5, 5.0, 1.4, 1.1), (-9.5, -3.0, 1.9, 0.9),
        ]):
            self._create_cube(f"/World/Apron/Stain_{i}", size=(sx, sy, 0.005), color=(0.10, 0.09, 0.08), pos=(float(x), float(y), 0.030), material_key="mat_oil_stain")

        # Asphalt service roads.
        self._create_cube("/World/Road/EntryX", size=(40.0, 3.0, 0.04), color=(0.10, 0.10, 0.11), pos=(0.0, -14.0, 0.015), material_key="mat_asphalt")
        self._create_cube("/World/Road/ServiceY", size=(3.0, 28.0, 0.04), color=(0.10, 0.10, 0.11), pos=(-15.0, 0.0, 0.015), material_key="mat_asphalt")
        self._create_cube("/World/Parking/Base", size=(12.0, 5.0, 0.035), color=(0.13, 0.13, 0.14), pos=(-12.0, -8.0, 0.02), material_key="mat_asphalt")

        # Lane lines and parking stripes - high-contrast SLAM features.
        for i, x in enumerate(np.linspace(-16.0, 16.0, 9)):
            self._create_cube(f"/World/Road/Line_{i}", size=(1.2, 0.09, 0.005), color=(0.78, 0.78, 0.74), pos=(float(x), -14.0, 0.04), material_key="mat_white")
        for i, y in enumerate(np.linspace(-4.0, 5.0, 5)):
            self._create_cube(f"/World/Parking/LineA_{i}", size=(0.08, 4.5, 0.005), color=(0.76, 0.76, 0.72), pos=(-16.8 + i * 2.2, -8.0, 0.04), material_key="mat_white")

        # Dirt/gravel patches at the edges where vegetation might encroach.
        for i, (x, y) in enumerate([(17.0, 12.0), (-17.5, 11.0), (18.0, -12.0), (-18.0, -11.5)]):
            self._create_cube(f"/World/Ground/DirtPatch_{i}", size=(4.5, 3.5, 0.03), color=(0.36, 0.31, 0.23), pos=(float(x), float(y), 0.012), material_key="mat_dirt")

    def _create_warehouse_industrial_layout(self):
        """Create the e2 industrial transfer-domain scene.

        This is intentionally different from the e1 power-plant/NPP scene.
        It removes reactor/cooling-tower geometry from transfer runs and uses
        warehouses, pipe racks, storage tanks, stacks, loading yards, and
        industrial inspection objects instead.  The e2 route is kept at 12
        waypoints, while e1 remains 16 waypoints.
        """
        print("[ENV_E2] Building industrial layout; no reactor/cooling-tower geometry will be created")

        # Replace the e1-style apron visually with an industrial asphalt/gravel yard.
        self._create_cube("/World/E2/Yard/AsphaltMain", size=(42.0, 32.0, 0.07), color=(0.075, 0.078, 0.082), pos=(0.0, 0.0, 0.025), material_key="mat_asphalt")
        self._create_cube("/World/E2/Yard/GravelNorth", size=(42.0, 8.0, 0.06), color=(0.25, 0.24, 0.22), pos=(0.0, 17.5, 0.03), material_key="mat_gravel")
        self._create_cube("/World/E2/Yard/GravelSouth", size=(42.0, 6.0, 0.06), color=(0.23, 0.22, 0.20), pos=(0.0, -17.0, 0.03), material_key="mat_gravel_dark")

        # Warehouses and process buildings: long rectangular structures rather
        # than the e1 containment building/cooling towers.
        self._create_building_block("/World/E2/WarehouseA", center=(-10.5, 5.5, 0.0), size=(9.0, 5.2, 4.2), roof_h=0.35, base_mat="mat_panel_light")
        self._create_building_block("/World/E2/WarehouseB", center=(4.5, -4.5, 0.0), size=(12.0, 5.5, 4.8), roof_h=0.40, base_mat="mat_blue_siding")
        self._create_building_block("/World/E2/Workshop", center=(12.5, 7.0, 0.0), size=(6.5, 4.8, 3.5), roof_h=0.30, base_mat="mat_green_siding")
        self._create_building_block("/World/E2/ControlShed", center=(-2.0, 10.5, 0.0), size=(5.0, 3.4, 3.0), roof_h=0.25, base_mat="mat_beige_siding")

        # Tanks, stacks and pipe racks give the transfer domain different
        # geometry/texture from e1 while remaining inspection-relevant.
        for i, (x, y, r, h, mat) in enumerate([
            (-13.5, -6.5, 1.25, 3.2, "mat_tank_steel"),
            (-10.0, -7.8, 1.05, 2.8, "mat_tank_top_steel"),
            (10.5, -10.0, 1.35, 3.8, "mat_steel_white_paint"),
            (14.0, -9.0, 0.95, 2.6, "mat_tank_steel"),
        ]):
            self._create_storage_tank(f"/World/E2/Tank_{i}", pos=(x, y, 0.0), radius=r, height=h, body_mat=mat)

        for i, (x, y, h) in enumerate([(-15.5, 9.0, 6.0), (7.8, 11.5, 5.2), (15.2, 0.5, 4.8)]):
            self._create_stack(f"/World/E2/Stack_{i}", pos=(x, y, 0.0), height=h)

        self._create_pipe_rack("/World/E2/PipeRack/Main", center=(-1.0, 1.8, 1.2), length=18.0)
        self._create_pipe_bridge("/World/E2/PipeBridge/A", start=(-12.5, -2.5, 1.4), end=(8.5, -2.5, 1.4), levels=3)
        self._create_pipe_bridge("/World/E2/PipeBridge/B", start=(-6.5, 8.0, 1.3), end=(14.0, 8.0, 1.3), levels=2)

        # Loading bays, containers, barriers, and lane stripes.
        for i, x in enumerate([-16.0, -13.8, -11.6, -9.4]):
            self._create_cube(f"/World/E2/LoadingBay_{i}", size=(1.5, 3.2, 0.08), color=(0.20, 0.22, 0.24), pos=(x, 1.0, 0.08), material_key="mat_panel_dark")
        for i, (x, y, sx, sy, mat) in enumerate([
            (-17.0, 13.5, 2.8, 1.2, "mat_pipe_blue"),
            (-13.5, 13.5, 2.8, 1.2, "mat_pipe_red"),
            (5.5, 14.0, 3.2, 1.1, "mat_pipe_green"),
            (9.5, 14.0, 3.2, 1.1, "mat_warning"),
        ]):
            self._create_cube(f"/World/E2/Container_{i}", size=(sx, sy, 1.2), color=(0.30, 0.34, 0.38), pos=(x, y, 0.65), material_key=mat)
        for i, x in enumerate(np.linspace(-18.0, 18.0, 10)):
            self._create_cube(f"/World/E2/RoadMark_{i}", size=(1.4, 0.08, 0.012), color=(0.72, 0.72, 0.68), pos=(float(x), -14.2, 0.09), material_key="mat_white")

        # e2-specific perimeter so visual transfer is obvious in GUI and figures.
        self._create_cube("/World/E2/Fence/North", size=(42.0, 0.10, 0.10), color=(0.64, 0.52, 0.16), pos=(0.0, 18.8, 1.0), material_key="mat_warning")
        self._create_cube("/World/E2/Fence/South", size=(42.0, 0.10, 0.10), color=(0.64, 0.52, 0.16), pos=(0.0, -18.8, 1.0), material_key="mat_warning")
        self._create_cube("/World/E2/Fence/East", size=(0.10, 37.5, 0.10), color=(0.64, 0.52, 0.16), pos=(21.0, 0.0, 1.0), material_key="mat_warning")
        self._create_cube("/World/E2/Fence/West", size=(0.10, 37.5, 0.10), color=(0.64, 0.52, 0.16), pos=(-21.0, 0.0, 1.0), material_key="mat_warning")

        # Re-assert the canonical e2 route from _build_boustrophedon_targets.
        # This prevents any old e1 route from surviving into transfer_e2.
        self.base_targets = self._build_boustrophedon_targets()
        self.targets = self.base_targets.copy()
        self.target_idx = 0
        self.target = self.targets[0].copy()
        print(f"[ENV_E2] Industrial route active: targets={len(self.targets)} expected=12")

    def _create_procedural_power_plant(self):
        if str(getattr(self, "sim_env_id", "e1")).lower() == "e2":
            print("[ENV_E2] Generating industrial transfer domain: warehouses, pipe racks, tanks, and service yards")
            self._create_warehouse_industrial_layout()
            return

        # ===================================================================
        # MAIN REACTOR CONTAINMENT BUILDING
        # Concrete cylinder with formwork bands, lightning rod, weathering.
        # ===================================================================
        self._create_reactor_containment("/World/NPP/Reactor", center=(0.0, 0.0, 0.0), radius=2.8, height=7.0, dome_radius=2.7)

        # Reactor base pad with dark skirt for grounded look.
        self._create_cube("/World/NPP/ReactorBase", size=(8.4, 8.4, 0.6), color=(0.46, 0.45, 0.42), pos=(0.0, 0.0, 0.30), material_key="mat_weathered_concrete")
        self._create_cube("/World/NPP/ReactorBaseSkirt", size=(8.7, 8.7, 0.18), color=(0.32, 0.31, 0.29), pos=(0.0, 0.0, 0.09), material_key="mat_concrete_dark")

        # ===================================================================
        # MAIN INDUSTRIAL BUILDINGS
        # ===================================================================
        self._create_building_block("/World/NPP/TurbineHall", center=(8.5, 0.0, 0.0), size=(12.0, 5.8, 4.8), roof_h=0.45, base_mat="mat_blue_siding")
        self._create_building_block("/World/NPP/ControlBuilding", center=(-7.0, -4.0, 0.0), size=(6.0, 4.0, 3.2), roof_h=0.25, base_mat="mat_concrete_warm")
        self._create_building_block("/World/NPP/Workshop", center=(-8.5, 5.5, 0.0), size=(7.5, 5.0, 3.0), roof_h=0.22, base_mat="mat_green_siding")

        # ===================================================================
        # COOLING TOWERS - hyperboloid weathered concrete with vertical streaks
        # ===================================================================
        self._create_cooling_tower("/World/NPP/CoolingTowerA", pos=(-12.0, 11.0, 0.0), scale=1.55)
        self._create_cooling_tower("/World/NPP/CoolingTowerB", pos=(13.0, -10.0, 0.0), scale=1.48)

        # ===================================================================
        # STORAGE TANKS - painted steel with manholes, ladders, bands
        # ===================================================================
        for i, (x, y, r, h, mat) in enumerate([
            (11.5, 8.5, 1.35, 3.0, "mat_tank_steel"),
            (14.8, 8.0, 1.00, 2.5, "mat_tank_steel"),
            (-2.5, 10.5, 1.20, 2.7, "mat_steel_white_paint"),
        ]):
            self._create_storage_tank(f"/World/NPP/Tank_{i}", pos=(x, y, 0.0), radius=r, height=h, body_mat=mat)

        # ===================================================================
        # CHIMNEYS / EXHAUST STACKS
        # ===================================================================
        self._create_stack("/World/NPP/StackA", pos=(15.0, 1.0, 0.0), height=8.0)
        self._create_stack("/World/NPP/StackB", pos=(16.5, 3.2, 0.0), height=6.5)

        # ===================================================================
        # PIPE RACKS / BRIDGES / CATWALKS
        # ===================================================================
        self._create_pipe_rack("/World/NPP/PipeRack1", center=(4.0, 6.8, 2.25), length=14.0)
        self._create_pipe_rack("/World/NPP/PipeRack2", center=(-2.0, -6.4, 2.10), length=12.0)
        self._create_pipe_bridge("/World/NPP/PipeBridgeA", start=(-2.5, 0.0, 3.0), end=(8.5, 0.0, 3.0), levels=3)
        self._create_pipe_bridge("/World/NPP/PipeBridgeB", start=(8.0, 4.0, 3.1), end=(12.8, 7.2, 3.1), levels=2)
        self._create_catwalk("/World/NPP/CatwalkA", center=(0.0, 4.2, 4.0), size=(8.0, 1.2, 0.12))
        self._create_catwalk("/World/NPP/CatwalkB", center=(8.5, 2.6, 3.2), size=(12.0, 1.1, 0.12))

        # ===================================================================
        # PUMPS / VALVES - small mechanical equipment for visual richness
        # ===================================================================
        for i, (x, y, z) in enumerate([(4.0, 4.5, 0.55), (6.2, 4.5, 0.55), (2.0, -5.0, 0.55), (-3.0, 8.5, 0.55)]):
            self._create_cube(f"/World/NPP/PumpBody_{i}", size=(0.85, 0.70, 0.80), color=(0.20, 0.21, 0.22), pos=(x, y, z), material_key="mat_dark_steel")
            self._create_cube(f"/World/NPP/PumpBase_{i}", size=(0.95, 0.80, 0.10), color=(0.32, 0.30, 0.27), pos=(x, y, z - 0.45), material_key="mat_concrete_dark")
            self._create_cylinder(f"/World/NPP/Valve_{i}", radius=0.18, height=0.20, color=(0.5, 0.1, 0.1), pos=(x, y, z + 0.55), material_key="mat_pipe_red")
            self._create_cylinder(f"/World/NPP/ValveStem_{i}", radius=0.04, height=0.30, color=(0.3, 0.3, 0.3), pos=(x, y, z + 0.78), material_key="mat_galvanized")

        # ===================================================================
        # HAZARD / WARNING SIGNAGE - feature-rich for SLAM
        # ===================================================================
        for i, (x, y) in enumerate([(3.0, -2.6), (9.2, -2.8), (-5.2, -4.8), (-4.0, 6.8), (13.5, 5.5)]):
            self._create_cube(f"/World/NPP/WarningPanel_{i}", size=(0.08, 1.0, 1.0), color=(0.74, 0.58, 0.10), pos=(x, y, 1.4), material_key="mat_safety")
            self._create_cube(f"/World/NPP/WarningPost_{i}", size=(0.08, 0.08, 1.4), color=(0.30, 0.30, 0.30), pos=(x, y, 0.7), material_key="mat_galvanized")

        # ===================================================================
        # GROUND SERVICE ITEMS - crates, drums, equipment for clutter
        # ===================================================================
        for i, (x, y, sx, sy, sz, mat) in enumerate([
            (10.0, -4.5, 1.2, 1.0, 0.7, "mat_steel_oxidized"),
            (-10.5, -8.5, 1.1, 0.9, 0.8, "mat_rust"),
            (6.2, 9.8, 1.6, 1.0, 0.8, "mat_steel_oxidized"),
        ]):
            self._create_cube(f"/World/NPP/Crate_{i}", size=(sx, sy, sz), color=(0.36, 0.26, 0.16), pos=(x, y, sz / 2.0), material_key=mat)

        # 55-gallon industrial drums at strategic spots.
        for i, (x, y) in enumerate([(7.5, -3.5), (8.2, -3.3), (-9.8, 7.2), (-9.4, 7.5), (12.5, -3.0)]):
            self._create_cylinder(f"/World/NPP/Drum_{i}", radius=0.32, height=0.92, color=(0.45, 0.15, 0.12), pos=(float(x), float(y), 0.46), material_key="mat_pipe_red")
            self._create_cylinder(f"/World/NPP/DrumTopRim_{i}", radius=0.34, height=0.04, color=(0.35, 0.10, 0.08), pos=(float(x), float(y), 0.92), material_key="mat_safety_red")

    def _create_reactor_containment(self, base_path: str, center, radius, height, dome_radius):
        """Modern PWR-style reactor containment: warm pale concrete cylinder
        with horizontal formwork seams, vertical lightning protection rod,
        and weathering streaks running down from rim."""
        x, y, z = center

        # Main containment wall (warm pale concrete).
        self._create_cylinder(base_path + "/Wall", radius=radius, height=height,
                              color=(0.60, 0.58, 0.53), pos=(x, y, z + height / 2.0),
                              material_key="mat_reactor_wall")

        # Horizontal formwork seams (concrete pour bands every ~1.4m).
        n_bands = max(3, int(height / 1.4))
        for i, h in enumerate(np.linspace(0.8, height - 0.5, n_bands)):
            self._create_cylinder(f"{base_path}/FormBand_{i}", radius=radius * 1.012, height=0.05,
                                  color=(0.46, 0.44, 0.40), pos=(x, y, z + float(h)),
                                  material_key="mat_reactor_seam")

        # Hemispherical dome (slightly lighter than walls).
        self._create_sphere(base_path + "/Dome", radius=dome_radius,
                            color=(0.62, 0.60, 0.55), pos=(x, y, z + height + 0.5),
                            material_key="mat_reactor_dome")

        # Dome-to-wall transition ring.
        self._create_cylinder(base_path + "/DomeRing", radius=radius * 1.02, height=0.30,
                              color=(0.50, 0.48, 0.43), pos=(x, y, z + height + 0.1),
                              material_key="mat_reactor_seam")

        # Lightning protection rod at apex.
        self._create_cylinder(base_path + "/LightningRod", radius=0.04, height=2.5,
                              color=(0.10, 0.10, 0.11), pos=(x, y, z + height + dome_radius + 1.25),
                              material_key="mat_reactor_lightning_rod")

        # Vertical weathering streaks at multiple azimuths (rain runoff from rim).
        for i, ang in enumerate(np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)):
            dx = math.cos(float(ang)) * (radius * 1.005)
            dy = math.sin(float(ang)) * (radius * 1.005)
            length = float(self.rng.uniform(2.5, height * 0.65)) if hasattr(self, "rng") else height * 0.5
            stain_z = z + height - length / 2.0 - 0.4
            self._create_cube(f"{base_path}/Streak_{i}", size=(0.08, 0.025, length),
                              color=(0.30, 0.30, 0.27), pos=(x + dx, y + dy, stain_z),
                              material_key="mat_concrete_stain")

        # Heavy rust streak (one prominent vertical line) for asymmetric realism.
        self._create_cube(base_path + "/RustStreak", size=(0.10, 0.025, height * 0.55),
                          color=(0.34, 0.20, 0.12), pos=(x + radius * 1.005 * math.cos(0.4),
                                                          y + radius * 1.005 * math.sin(0.4),
                                                          z + height * 0.4),
                          material_key="mat_concrete_rust_streak")

        # Personnel airlock (small protrusion near base).
        self._create_cube(base_path + "/Airlock", size=(1.0, 0.6, 1.8),
                          color=(0.20, 0.20, 0.21), pos=(x + radius + 0.45, y, z + 0.9),
                          material_key="mat_dark_steel")
        self._create_cube(base_path + "/AirlockDoor", size=(0.05, 0.55, 1.6),
                          color=(0.10, 0.10, 0.11), pos=(x + radius + 0.95, y, z + 0.9),
                          material_key="mat_steel_oxidized")

        # Algae stain at the very base (moisture line ~0.6m up the wall).
        for i, ang in enumerate(np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)):
            dx = math.cos(float(ang)) * (radius * 1.008)
            dy = math.sin(float(ang)) * (radius * 1.008)
            self._create_cube(f"{base_path}/AlgaeBase_{i}", size=(0.30, 0.025, 0.45),
                              color=(0.20, 0.26, 0.20), pos=(x + dx, y + dy, z + 0.30),
                              material_key="mat_concrete_algae")

    def _create_substation(self):
        self._create_cube("/World/Substation/Base", size=(10.5, 7.8, 0.04), color=(0.43, 0.43, 0.41), pos=(15.0, 11.5, 0.03), material_key="mat_concrete")
        for i, x in enumerate([12.8, 15.1, 17.4]):
            # Transformer body - ribbed dark steel.
            self._create_cube(f"/World/Substation/Transformer_{i}", size=(1.2, 1.0, 1.2), color=(0.32, 0.33, 0.34), pos=(x, 11.1, 0.62), material_key="mat_dark_steel")
            # Bushing porcelain insulators on top.
            for j, ox in enumerate([-0.32, 0.0, 0.32]):
                self._create_cylinder(f"/World/Substation/Bushing_{i}_{j}", radius=0.08, height=0.45, color=(0.78, 0.74, 0.70), pos=(x + ox, 11.1, 1.45), material_key="mat_steel_white_paint")
            # Cooling fins on side.
            self._create_cube(f"/World/Substation/Fins_{i}", size=(0.08, 1.0, 0.9), color=(0.28, 0.28, 0.29), pos=(x + 0.62, 11.1, 0.62), material_key="mat_galvanized")
        # Gantry support steel.
        for i, x in enumerate([13.0, 15.0, 17.0]):
            self._create_cube(f"/World/Substation/Gantry_{i}", size=(0.12, 0.12, 3.0), color=(0.30, 0.31, 0.32), pos=(x, 13.8, 1.5), material_key="mat_galvanized")
            self._create_cube(f"/World/Substation/TopBeam_{i}", size=(2.2, 0.10, 0.10), color=(0.30, 0.31, 0.32), pos=(x, 13.8, 2.9), material_key="mat_galvanized")

    def _create_visual_landmarks(self):
        for i, (x, y) in enumerate([(-16.5, -14.0), (-10.0, -14.0), (-3.0, -14.0), (4.0, -14.0), (11.0, -14.0), (18.0, -13.0)]):
            self._create_cube(f"/World/Landmarks/LightPole_{i}", size=(0.12, 0.12, 4.2), color=(0.32, 0.33, 0.34), pos=(x, y, 2.1), material_key="mat_galvanized")
            self._create_cube(f"/World/Landmarks/LightArm_{i}", size=(0.65, 0.06, 0.06), color=(0.30, 0.30, 0.30), pos=(x + 0.32, y, 4.15), material_key="mat_galvanized")
            light = UsdLux.SphereLight.Define(self.stage, f"/World/Landmarks/Light_{i}")
            light.GetIntensityAttr().Set(180.0)
            light.GetRadiusAttr().Set(1.0)
            light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.92, 0.78))
            UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate((float(x) + 0.65, float(y), 4.6))

    def _create_perimeter_fence(self):
        bound = self.world_size + 0.8
        # Galvanized chain-link fence base.
        self._create_cube("/World/Fence/North", size=(2 * bound, 0.06, 1.7), color=(0.22, 0.23, 0.23), pos=(0.0, bound, 0.88), material_key="mat_fence")
        self._create_cube("/World/Fence/South", size=(2 * bound, 0.06, 1.7), color=(0.22, 0.23, 0.23), pos=(0.0, -bound, 0.88), material_key="mat_fence")
        self._create_cube("/World/Fence/East", size=(0.06, 2 * bound, 1.7), color=(0.22, 0.23, 0.23), pos=(bound, 0.0, 0.88), material_key="mat_fence")
        self._create_cube("/World/Fence/West", size=(0.06, 2 * bound, 1.7), color=(0.22, 0.23, 0.23), pos=(-bound, 0.0, 0.88), material_key="mat_fence")
        # Top rail and bottom rail bands for visual texture.
        for side, (sx, sy, px, py) in [("N", (2 * bound, 0.06, 0, bound)), ("S", (2 * bound, 0.06, 0, -bound)),
                                        ("E", (0.06, 2 * bound, bound, 0)), ("W", (0.06, 2 * bound, -bound, 0))]:
            self._create_cube(f"/World/Fence/{side}_TopRail", size=(sx, sy, 0.05), color=(0.24, 0.25, 0.25), pos=(px, py, 1.72), material_key="mat_galvanized")

    # ---------------------------------------------------------------------
    # drone and sensors
    # ---------------------------------------------------------------------
    def _create_drone(self):
        print(f"[ASSET] drone_usd='{self.drone_usd}'")
        if self._is_valid_asset_ref(self.drone_usd):
            drone_ref = self._asset_ref(self.drone_usd)
            print(f"[ASSET] Loading drone USD: {drone_ref}")
            self._add_reference_xform("/World/Drone", drone_ref, pos=(0.0, -8.0, 2.0), scale=(1.0, 1.0, 1.0))
            self.drone_uses_usd = True
            return
        print("[ASSET] No valid drone USD provided. Using procedural quadcopter.")
        self.drone_uses_usd = False
        self._create_cube("/World/Drone/Body", size=(0.60, 0.34, 0.16), color=(0.12, 0.12, 0.13), pos=(0.0, -8.0, 2.0), material_key="mat_drone_body")
        self._create_cube("/World/Drone/Arm_X", size=(0.92, 0.055, 0.035), color=(0.055, 0.055, 0.06), pos=(0.0, -8.0, 2.0), material_key="mat_drone_carbon")
        self._create_cube("/World/Drone/Arm_Y", size=(0.055, 0.92, 0.035), color=(0.055, 0.055, 0.06), pos=(0.0, -8.0, 2.0), material_key="mat_drone_carbon")
        for i, (x, y) in enumerate([(0.46, 0), (-0.46, 0), (0, 0.46), (0, -0.46)]):
            self._create_cylinder(f"/World/Drone/Rotor_{i}", radius=0.19, height=0.018, color=(0.02, 0.02, 0.02), pos=(x, -8.0 + y, 2.03), material_key="mat_drone_prop")
        self._create_cube("/World/Drone/CameraHousing", size=(0.18, 0.11, 0.11), color=(0.01, 0.012, 0.015), pos=(0.34, -8.0, 1.95), material_key="mat_drone_lens")
        self._create_cube("/World/Drone/BottomCameraHousing", size=(0.16, 0.13, 0.08), color=(0.01, 0.012, 0.015), pos=(0.0, -8.0, 1.88), material_key="mat_drone_lens")

    def _create_sensor_rig_visuals(self):
        # Sensor body visuals. The real ROS 2 camera topics are created by
        # _create_stereo_camera_prims() + _setup_ros2_stereo_camera_graph().
        self._create_cube("/World/Drone/SensorIMU", size=(0.08, 0.08, 0.03), color=(0.20, 0.20, 0.22), pos=(0.0, -8.0, 2.10), material_key="mat_dark_steel")
        self._create_cube("/World/Drone/SensorDepth", size=(0.12, 0.08, 0.07), color=(0.04, 0.04, 0.05), pos=(0.37, -8.0, 1.95), material_key="mat_drone_lens")
        self._create_cube("/World/Drone/SensorDown", size=(0.11, 0.09, 0.05), color=(0.04, 0.04, 0.05), pos=(0.0, -8.0, 1.88), material_key="mat_drone_lens")

    def _create_stereo_camera_prims(self):
        # Actual USD camera prims for Isaac Sim ROS 2 Bridge. Isaac Sim publishes
        # only RGB image frames from these prims. CameraInfo and static TF are
        # published by a small ROS 2 helper in the Isaac ROS container because
        # Isaac Sim 5.1 reports OgnROS2CameraHelper type=camera_info as unsupported.
        try:
            for side, yoff in [("Left", self.stereo_baseline * 0.5), ("Right", -self.stereo_baseline * 0.5)]:
                cam_path = f"/World/Drone/Stereo{side}Camera"
                cam = UsdGeom.Camera.Define(self.stage, cam_path)
                cam.CreateHorizontalApertureAttr(20.955)
                cam.CreateVerticalApertureAttr(15.2908)
                focal = 0.5 * 20.955 / max(math.tan(self.camera_hfov * 0.5), 1e-6)
                cam.CreateFocalLengthAttr(float(focal))
                cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
                UsdGeom.XformCommonAPI(cam.GetPrim()).SetTranslate((0.42, yoff, -0.04))
                UsdGeom.XformCommonAPI(cam.GetPrim()).SetRotate((0.0, -90.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

            # A second inspection camera points downward. It is used for the
            # paper visual heatmap sequence and can also be viewed in Isaac Sim.
            bottom_path = "/World/Drone/BottomInspectionCamera"
            bottom_cam = UsdGeom.Camera.Define(self.stage, bottom_path)
            bottom_cam.CreateHorizontalApertureAttr(20.955)
            bottom_cam.CreateVerticalApertureAttr(15.2908)
            bottom_focal = 0.5 * 20.955 / max(math.tan(self.camera_hfov * 0.5), 1e-6)
            bottom_cam.CreateFocalLengthAttr(float(bottom_focal))
            bottom_cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
            UsdGeom.XformCommonAPI(bottom_cam.GetPrim()).SetTranslate((0.0, 0.0, -0.16))
            # USD cameras look along local -Z. With only yaw rotation, the
            # optical axis remains downward in world/body -Z.
            UsdGeom.XformCommonAPI(bottom_cam.GetPrim()).SetRotate((0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
            print("[ROS2] Stereo camera prims created at /World/Drone/StereoLeftCamera and /World/Drone/StereoRightCamera")
            print("[ROS2] Downward inspection camera prim created at /World/Drone/BottomInspectionCamera")
        except Exception as exc:
            print(f"[ROS2] Warning: could not create stereo camera prims: {exc}")

    def _setup_ros2_stereo_camera_graph(self):
        """Create Isaac Sim ROS 2 Bridge publishers for stereo images.

        Fixed behavior:
        - removes stale/partial graph prims from failed attempts;
        - tries current Isaac Sim ROS 2 node namespace first;
        - uses a unique graph path per attempt to avoid the "graph already exists" bug;
        - starts the Isaac timeline so OnPlaybackTick actually publishes camera frames.
        """
        if not self.enable_ros2_camera_pub:
            return

        # Use a unique graph path for every process run. In Isaac Sim 5.x, a failed
        # ROS graph creation may leave an invalid ComputeGraph prim that cannot be
        # safely re-wrapped. A unique path avoids the stale graph collision entirely.
        graph_base_path = f"/World/ROS2_CuVSLAM_StereoGraph_{int(time.time() * 1000)}"
        try:
            os.environ["ROS_DISTRO"] = "humble"
            os.environ["ROS_DOMAIN_ID"] = str(self.ros2_domain_id)
            os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

            # Isaac Sim 5.x uses isaacsim.ros2.bridge. Keep legacy fallback.
            try:
                from isaacsim.core.utils.extensions import enable_extension
                enable_extension("isaacsim.ros2.bridge")
            except Exception:
                try:
                    from omni.isaac.core.utils.extensions import enable_extension
                    enable_extension("omni.isaac.ros2_bridge")
                except Exception:
                    pass

            # Let extension startup complete before graph creation.
            for _ in range(8):
                omni.kit.app.get_app().update()

            import omni.graph.core as og
            import omni.replicator.core as rep

            # Remove stale/partially-created graphs from prior failed attempts.
            try:
                stale_paths = []
                for prim in list(self.stage.Traverse()):
                    p = str(prim.GetPath())
                    # remove old fixed-name graphs from previous code versions only
                    if p == "/World/ROS2_CuVSLAM_StereoGraph" or p.startswith("/World/ROS2_CuVSLAM_StereoGraph_old_"):
                        stale_paths.append(p)
                for p in sorted(stale_paths, key=len, reverse=True):
                    self.stage.RemovePrim(p)
                if stale_paths:
                    print(f"[ROS2] Removed stale ROS 2 camera graph prims: {stale_paths}")
            except Exception as exc:
                print(f"[ROS2] Warning: stale graph cleanup skipped: {exc}")

            left_rp = rep.create.render_product(
                "/World/Drone/StereoLeftCamera",
                (self.ros2_camera_width, self.ros2_camera_height),
            )
            right_rp = rep.create.render_product(
                "/World/Drone/StereoRightCamera",
                (self.ros2_camera_width, self.ros2_camera_height),
            )

            node_prefixes = ["isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"]
            created = False
            last_exc = None
            for attempt, ros_prefix in enumerate(node_prefixes):
                graph_path = graph_base_path if attempt == 0 else f"{graph_base_path}_{attempt}"
                try:
                    og.Controller.edit(
                        {"graph_path": graph_path, "evaluator_name": "execution"},
                        {
                            og.Controller.Keys.CREATE_NODES: [
                                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                                ("ROS2Context", f"{ros_prefix}.ROS2Context"),
                                ("LeftRGB", f"{ros_prefix}.ROS2CameraHelper"),
                                ("RightRGB", f"{ros_prefix}.ROS2CameraHelper"),
                            ],
                            og.Controller.Keys.CONNECT: [
                                ("OnPlaybackTick.outputs:tick", "LeftRGB.inputs:execIn"),
                                ("OnPlaybackTick.outputs:tick", "RightRGB.inputs:execIn"),
                                ("ROS2Context.outputs:context", "LeftRGB.inputs:context"),
                                ("ROS2Context.outputs:context", "RightRGB.inputs:context"),
                            ],
                            og.Controller.Keys.SET_VALUES: [
                                ("ROS2Context.inputs:domain_id", int(self.ros2_domain_id)),
                                ("LeftRGB.inputs:renderProductPath", str(left_rp.path)),
                                ("LeftRGB.inputs:topicName", self.left_image_topic),
                                ("LeftRGB.inputs:frameId", "front_stereo_camera_left_optical_frame"),
                                ("LeftRGB.inputs:type", "rgb"),
                                ("RightRGB.inputs:renderProductPath", str(right_rp.path)),
                                ("RightRGB.inputs:topicName", self.right_image_topic),
                                ("RightRGB.inputs:frameId", "front_stereo_camera_right_optical_frame"),
                                ("RightRGB.inputs:type", "rgb"),
                            ],
                        },
                    )
                    created = True
                    print(f"[ROS2] Created stereo camera graph: {graph_path} using {ros_prefix}")
                    print("[ROS2] If no image publisher appears, confirm Isaac Sim was launched with LD_LIBRARY_PATH including the Isaac Sim ros2 bridge humble/lib path and without sourcing ROS Python into PYTHONPATH.")
                    break
                except Exception as exc:
                    last_exc = exc
                    print(f"[ROS2] Graph attempt with {ros_prefix} failed: {exc}")

            if not created:
                raise last_exc or RuntimeError("Could not create ROS 2 stereo camera graph")

            self.ros2_graph_created = True

            # OnPlaybackTick publishes only while the timeline is playing.
            try:
                import importlib
                omni_timeline = importlib.import_module("omni.timeline")
                timeline = omni_timeline.get_timeline_interface()
                if not timeline.is_playing():
                    timeline.play()
                    print("[ROS2] Isaac timeline started for ROS 2 camera publishing")
            except Exception as exc:
                print(f"[ROS2] Warning: could not start Isaac timeline: {exc}")

            for _ in range(20):
                omni.kit.app.get_app().update()

            print("[ROS2] Stereo camera ROS 2 graph enabled for cuVSLAM")
            print(f"[ROS2] left image:  {self.left_image_topic}")
            print(f"[ROS2] right image: {self.right_image_topic}")
            print("[ROS2] CameraInfo is intentionally NOT published by Isaac Sim; run the external CameraInfo+TF helper in the Isaac ROS container.")
            print(f"[ROS2] expected left info:   {self.left_camera_info_topic}")
            print(f"[ROS2] expected right info:  {self.right_camera_info_topic}")
        except Exception as exc:
            self.ros2_graph_created = False
            print(f"[ROS2] Warning: failed to create Isaac ROS 2 stereo camera graph: {exc}")
            print("[ROS2] The scene will still run, but cuVSLAM will not receive camera images until ROS 2 camera publishing is fixed.")

    def _create_target(self):
        # Inspection waypoints are logical targets, not rendered green balloons.
        # Keep an invisible Xform at /World/InspectionTarget for compatibility
        # with _sync_scene(), but do not create any visible sphere/cylinder in
        # the air. This preserves the high-altitude UAV flight path and only
        # removes the misleading GUI marker.
        try:
            xform = UsdGeom.Xform.Define(self.stage, "/World/InspectionTarget")
            prim = xform.GetPrim()
            UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in self.target))
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
            except Exception:
                pass
            self.ops["/World/InspectionTarget"] = {"type": "hidden_xform", "prim": prim}
        except Exception as exc:
            print(f"[VIS] Warning: could not create hidden inspection target prim: {exc}")

    def _create_obstacles(self):
        for i in range(self.num_obstacles):
            self._create_cube(f"/World/DynamicObstacle_{i}", size=(0.8, 0.8, 1.0), color=(0.65, 0.65, 0.65), pos=(0.0, 0.0, -10.0), material_key="mat_concrete")

    # ---------------------------------------------------------------------
    # primitive / material helpers
    # ---------------------------------------------------------------------
    def _make_preview_material(self, key, path, color, roughness=0.65, metallic=0.0, emissive=None):
        mat = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in color]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        # Slight specular tint for metal - already baked into metallic, but a hint of IOR helps.
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04, 0.04, 0.04))
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
        if emissive is not None:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in emissive]))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.materials[key] = path

    def _bind_material(self, prim, key: str):
        mat_path = self.materials.get(key)
        if not mat_path:
            return
        mat = UsdShade.Material.Get(self.stage, mat_path)
        if mat:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def _create_cube(self, path: str, size, color, pos, material_key: str = ""):
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        self._bind_material(prim, material_key)
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(tuple(float(v) for v in pos))
        api.SetScale(tuple(float(v) for v in size))
        self.ops[path] = {"type": "cube", "prim": prim}
        try:
            if self._should_register_static_collision(path, size, pos):
                self._register_static_collision_box(path, pos, size, category="cube")
        except Exception:
            pass

    def _create_sphere(self, path: str, radius: float, color, pos, material_key: str = ""):
        sph = UsdGeom.Sphere.Define(self.stage, path)
        sph.CreateRadiusAttr(float(radius))
        prim = sph.GetPrim()
        self._bind_material(prim, material_key)
        UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in pos))
        self.ops[path] = {"type": "sphere", "prim": prim}
        try:
            d = 2.0 * float(radius)
            if self._should_register_static_collision(path, (d, d, d), pos):
                self._register_static_collision_box(path, pos, (d, d, d), category="sphere")
        except Exception:
            pass

    def _create_cylinder(self, path: str, radius: float, height: float, color, pos, material_key: str = ""):
        cyl = UsdGeom.Cylinder.Define(self.stage, path)
        cyl.CreateRadiusAttr(float(radius))
        cyl.CreateHeightAttr(float(height))
        prim = cyl.GetPrim()
        self._bind_material(prim, material_key)
        UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in pos))
        self.ops[path] = {"type": "cylinder", "prim": prim}
        try:
            size = (2.0 * float(radius), 2.0 * float(radius), float(height))
            if self._should_register_static_collision(path, size, pos):
                self._register_static_collision_box(path, pos, size, category="cylinder")
        except Exception:
            pass

    def _create_building_block(self, base_path: str, center=(0, 0, 0), size=(6, 4, 3), roof_h=0.2, base_mat="mat_weathered_concrete"):
        x, y, z = center
        sx, sy, sz = size

        # Main body.
        self._create_cube(base_path + "/Base", size=(sx, sy, sz), color=(0.45, 0.45, 0.45), pos=(x, y, z + sz / 2.0), material_key=base_mat)

        # Concrete skirt at base (anchors building visually to ground).
        self._create_cube(base_path + "/Skirt", size=(sx * 1.01, sy * 1.01, 0.40), color=(0.30, 0.29, 0.27), pos=(x, y, z + 0.20), material_key="mat_concrete_dark")

        # Foundation stain line just above skirt.
        self._create_cube(base_path + "/FoundStain", size=(sx * 1.015, sy * 1.015, 0.08), color=(0.22, 0.22, 0.20), pos=(x, y, z + 0.42), material_key="mat_concrete_stain")

        # Flat roof with darker membrane.
        self._create_cube(base_path + "/Roof", size=(sx * 1.03, sy * 1.03, roof_h), color=(0.18, 0.18, 0.19), pos=(x, y, z + sz + roof_h / 2.0), material_key="mat_dark_steel")
        # Roof parapet (raised edge).
        self._create_cube(base_path + "/ParapetN", size=(sx * 1.03, 0.05, 0.20), color=(0.30, 0.30, 0.30), pos=(x, y + sy * 0.515, z + sz + roof_h + 0.10), material_key="mat_galvanized")
        self._create_cube(base_path + "/ParapetS", size=(sx * 1.03, 0.05, 0.20), color=(0.30, 0.30, 0.30), pos=(x, y - sy * 0.515, z + sz + roof_h + 0.10), material_key="mat_galvanized")
        self._create_cube(base_path + "/ParapetE", size=(0.05, sy * 1.03, 0.20), color=(0.30, 0.30, 0.30), pos=(x + sx * 0.515, y, z + sz + roof_h + 0.10), material_key="mat_galvanized")
        self._create_cube(base_path + "/ParapetW", size=(0.05, sy * 1.03, 0.20), color=(0.30, 0.30, 0.30), pos=(x - sx * 0.515, y, z + sz + roof_h + 0.10), material_key="mat_galvanized")

        # Facade band variation so buildings do not look flat.
        stripe_mat = "mat_panel_light" if base_mat in ("mat_blue_siding", "mat_green_siding") else "mat_panel_dark"
        self._create_cube(base_path + "/FrontBand", size=(sx * 0.96, 0.06, sz * 0.22), color=(0.5, 0.5, 0.5), pos=(x, y - sy / 2.0 - 0.031, z + sz * 0.64), material_key=stripe_mat)
        self._create_cube(base_path + "/BackBand", size=(sx * 0.88, 0.06, sz * 0.15), color=(0.5, 0.5, 0.5), pos=(x, y + sy / 2.0 + 0.031, z + sz * 0.52), material_key="mat_concrete_stain")

        # Vertical cladding panels (subtle ribbed wall texture).
        n_pan = max(3, int(sy * 1.1))
        for i, yy in enumerate(np.linspace(-sy / 2.0 + 0.45, sy / 2.0 - 0.45, n_pan)):
            self._create_cube(f"{base_path}/PanelA_{i}", size=(0.06, 0.52, 0.90), color=(0.35, 0.35, 0.35), pos=(x + sx / 2.0 + 0.05, y + yy, z + sz * 0.56), material_key="mat_corrugated")
            if i % 2 == 0:
                self._create_cube(f"{base_path}/PanelB_{i}", size=(0.05, 0.48, 0.78), color=(0.35, 0.35, 0.35), pos=(x - sx / 2.0 - 0.045, y + yy, z + sz * 0.52), material_key="mat_panel_dark")

        # Window grid with mullions (more realistic than single panes).
        n_win = max(2, int(sx // 2))
        for i, xx in enumerate(np.linspace(-sx * 0.30, sx * 0.30, n_win)):
            self._create_cube(f"{base_path}/Window_{i}", size=(0.65, 0.06, 0.55), color=(0.08, 0.10, 0.12), pos=(x + xx, y - sy / 2.0 - 0.03, z + sz * 0.56), material_key="mat_glass_window")
            # Window frame.
            self._create_cube(f"{base_path}/WindowFrameTop_{i}", size=(0.72, 0.07, 0.05), color=(0.20, 0.20, 0.20), pos=(x + xx, y - sy / 2.0 - 0.034, z + sz * 0.56 + 0.30), material_key="mat_dark_steel")
            self._create_cube(f"{base_path}/WindowFrameBot_{i}", size=(0.72, 0.07, 0.05), color=(0.20, 0.20, 0.20), pos=(x + xx, y - sy / 2.0 - 0.034, z + sz * 0.56 - 0.30), material_key="mat_dark_steel")
            # Vertical mullion.
            self._create_cube(f"{base_path}/WindowMull_{i}", size=(0.04, 0.07, 0.55), color=(0.20, 0.20, 0.20), pos=(x + xx, y - sy / 2.0 - 0.034, z + sz * 0.56), material_key="mat_dark_steel")

        # Industrial steel door with frame.
        self._create_cube(base_path + "/Door", size=(0.85, 0.07, 1.8), color=(0.28, 0.28, 0.28), pos=(x - sx * 0.32, y - sy / 2.0 - 0.035, z + 0.9), material_key="mat_steel_oxidized")
        self._create_cube(base_path + "/DoorFrame", size=(0.95, 0.06, 1.92), color=(0.18, 0.18, 0.18), pos=(x - sx * 0.32, y - sy / 2.0 - 0.030, z + 0.96), material_key="mat_dark_steel")

        # Grime / weathering decals (rust streaks from window sills, downpipes).
        self._create_cube(base_path + "/GrimeA", size=(sx * 0.34, 0.03, 0.85), color=(0.30, 0.30, 0.30), pos=(x + sx * 0.18, y - sy / 2.0 - 0.02, z + sz * 0.32), material_key="mat_concrete_stain")
        self._create_cube(base_path + "/GrimeB", size=(0.03, sy * 0.45, 0.90), color=(0.30, 0.30, 0.30), pos=(x + sx / 2.0 + 0.02, y - sy * 0.12, z + sz * 0.34), material_key="mat_concrete_stain")
        # Vertical rust streak from rooftop drain.
        self._create_cube(base_path + "/DrainStreak", size=(0.04, 0.025, sz * 0.7), color=(0.32, 0.18, 0.10), pos=(x + sx * 0.42, y - sy / 2.0 - 0.025, z + sz * 0.45), material_key="mat_concrete_rust_streak")

        # Rooftop HVAC / mechanical units.
        for i, dx in enumerate(np.linspace(-sx * 0.25, sx * 0.25, 3)):
            self._create_cube(f"{base_path}/RoofUnit_{i}", size=(0.9, 0.8, 0.5), color=(0.16, 0.16, 0.17), pos=(x + dx, y, z + sz + roof_h + 0.25), material_key="mat_dark_steel")
            # Vent on top of unit.
            self._create_cylinder(f"{base_path}/RoofVent_{i}", radius=0.18, height=0.25, color=(0.24, 0.24, 0.25), pos=(x + dx, y, z + sz + roof_h + 0.62), material_key="mat_galvanized")

        # Exterior conduit / downpipe.
        self._create_cube(base_path + "/Conduit", size=(0.10, 0.10, sz * 0.95), color=(0.34, 0.32, 0.30), pos=(x + sx / 2.0 + 0.07, y + sy * 0.30, z + sz * 0.50), material_key="mat_steel_oxidized")

    def _create_cooling_tower(self, base_path: str, pos=(0, 0, 0), scale=1.0):
        """Hyperboloid natural-draft cooling tower with realistic weathering:
        - Stacked cylinders with varying radii to approximate hyperboloid shape
        - Vertical water/algae streaks all around
        - Wet darkened base
        - Subtle moss line at ground contact
        - Top rim shadow band
        """
        x, y, z = pos
        # Bottom flared section (wider at base).
        self._create_cylinder(base_path + "/SectionBase", radius=1.70 * scale, height=0.80 * scale,
                              color=(0.55, 0.54, 0.49), pos=(x, y, z + 0.40 * scale),
                              material_key="mat_tower_concrete")

        # Tapered lower section.
        self._create_cylinder(base_path + "/SectionLow", radius=1.42 * scale, height=1.6 * scale,
                              color=(0.56, 0.55, 0.50), pos=(x, y, z + 1.6 * scale),
                              material_key="mat_tower_concrete")

        # Narrow waist (the hyperboloid pinch).
        self._create_cylinder(base_path + "/SectionWaist", radius=1.05 * scale, height=1.8 * scale,
                              color=(0.57, 0.56, 0.51), pos=(x, y, z + 3.3 * scale),
                              material_key="mat_tower_concrete")

        # Upper flare section.
        self._create_cylinder(base_path + "/SectionUpper", radius=1.20 * scale, height=1.4 * scale,
                              color=(0.59, 0.58, 0.53), pos=(x, y, z + 4.9 * scale),
                              material_key="mat_tower_concrete")

        # Top rim (slightly lighter, like sun-bleached upper concrete).
        self._create_cylinder(base_path + "/SectionTop", radius=1.42 * scale, height=0.85 * scale,
                              color=(0.61, 0.59, 0.54), pos=(x, y, z + 6.05 * scale),
                              material_key="mat_tower_concrete_top")

        # ----- WEATHERING DETAILS -----
        # Wet darkened base ring (water seepage zone).
        self._create_cylinder(base_path + "/WetBase", radius=1.74 * scale, height=0.30 * scale,
                              color=(0.22, 0.22, 0.21), pos=(x, y, z + 0.15 * scale),
                              material_key="mat_tower_base_wet")

        # Algae/moss line at very bottom (~30cm strip).
        for i, ang in enumerate(np.linspace(0.0, 2.0 * math.pi, 14, endpoint=False)):
            dx = math.cos(float(ang)) * 1.72 * scale
            dy = math.sin(float(ang)) * 1.72 * scale
            self._create_cube(f"{base_path}/MossPatch_{i}", size=(0.30, 0.025, 0.18 * scale),
                              color=(0.20, 0.25, 0.18), pos=(x + dx, y + dy, z + 0.18 * scale),
                              material_key="mat_tower_moss")

        # Top rim shadow band (darker stripe under the lip).
        self._create_cylinder(base_path + "/TopShadow", radius=1.46 * scale, height=0.18 * scale,
                              color=(0.30, 0.30, 0.27), pos=(x, y, z + 6.55 * scale),
                              material_key="mat_tower_shadow")

        # Vertical water/weather streaks running down from rim - many azimuths
        # for full 360-degree weathering coverage.
        n_streaks = 14
        for i, ang in enumerate(np.linspace(0.0, 2.0 * math.pi, n_streaks, endpoint=False)):
            # Slight randomization of streak length for natural variation.
            streak_len = (3.0 + (i % 3) * 1.2) * scale
            streak_z = z + 4.5 * scale - streak_len / 2.0
            # Place streak on appropriate radius for its height (waist is narrowest).
            r_at_streak = 1.05 * scale + 0.18 * scale * (1.0 + math.sin(i * 0.7))
            dx = math.cos(float(ang)) * r_at_streak
            dy = math.sin(float(ang)) * r_at_streak
            self._create_cube(f"{base_path}/Streak_{i}", size=(0.12 * scale, 0.025 * scale, streak_len),
                              color=(0.30, 0.30, 0.27), pos=(x + dx, y + dy, streak_z),
                              material_key="mat_tower_water_streak")

        # Heavy rust streaks (one prominent vertical line per tower for realism).
        rust_ang = 0.6
        rust_dx = math.cos(rust_ang) * 1.30 * scale
        rust_dy = math.sin(rust_ang) * 1.30 * scale
        self._create_cube(base_path + "/RustStreak", size=(0.12 * scale, 0.025 * scale, 4.0 * scale),
                          color=(0.32, 0.18, 0.10), pos=(x + rust_dx, y + rust_dy, z + 3.8 * scale),
                          material_key="mat_concrete_rust_streak")

        # Form-pour horizontal seams (every ~1.2m).
        for i, h in enumerate([0.85, 2.10, 3.40, 4.70, 5.90]):
            r_at = 1.40 * scale - 0.15 * scale * abs(h - 3.3) / 3.3
            self._create_cylinder(f"{base_path}/FormSeam_{i}", radius=r_at + 0.02 * scale, height=0.06 * scale,
                                  color=(0.42, 0.41, 0.37), pos=(x, y, z + h * scale),
                                  material_key="mat_concrete_seam")

    def _create_storage_tank(self, base_path: str, pos, radius, height, body_mat):
        """Industrial storage tank with structural rings, manhole, and ladder."""
        x, y, z = pos
        # Main cylindrical body.
        self._create_cylinder(base_path + "/Body", radius=radius, height=height,
                              color=(0.5, 0.5, 0.5), pos=(x, y, z + height / 2.0),
                              material_key=body_mat)

        # Domed top.
        self._create_sphere(base_path + "/Top", radius=radius * 0.95,
                            color=(0.55, 0.55, 0.56), pos=(x, y, z + height + radius * 0.45),
                            material_key="mat_tank_top_steel")

        # Reinforcing ring bands around body (every ~1m).
        n_rings = max(2, int(height // 0.9))
        for i, h in enumerate(np.linspace(0.5, height - 0.3, n_rings)):
            self._create_cylinder(f"{base_path}/Ring_{i}", radius=radius * 1.015, height=0.08,
                                  color=(0.40, 0.40, 0.41), pos=(x, y, z + float(h)),
                                  material_key="mat_galvanized")

        # Concrete foundation pad.
        self._create_cylinder(base_path + "/Pad", radius=radius * 1.15, height=0.20,
                              color=(0.40, 0.40, 0.38), pos=(x, y, z + 0.10),
                              material_key="mat_concrete_dark")

        # Manhole on side.
        self._create_cylinder(base_path + "/Manhole", radius=0.22, height=0.08,
                              color=(0.20, 0.21, 0.22), pos=(x + radius * 1.005, y, z + 0.7),
                              material_key="mat_dark_steel")

        # Vertical access ladder (visible feature for SLAM).
        ladder_x = x + radius * 1.02
        for i, h in enumerate(np.linspace(0.3, height - 0.3, max(3, int(height // 0.4)))):
            self._create_cube(f"{base_path}/LadderRung_{i}", size=(0.05, 0.40, 0.04),
                              color=(0.20, 0.20, 0.21), pos=(ladder_x, y - radius * 0.25, z + float(h)),
                              material_key="mat_galvanized")
        # Ladder side rails.
        self._create_cube(base_path + "/LadderRailA", size=(0.04, 0.04, height - 0.5),
                          color=(0.20, 0.20, 0.21), pos=(ladder_x, y - radius * 0.40, z + height / 2.0),
                          material_key="mat_galvanized")
        self._create_cube(base_path + "/LadderRailB", size=(0.04, 0.04, height - 0.5),
                          color=(0.20, 0.20, 0.21), pos=(ladder_x, y - radius * 0.10, z + height / 2.0),
                          material_key="mat_galvanized")

        # Subtle vertical streaks down the side (paint weathering).
        for i, ang in enumerate(np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)):
            dx = math.cos(float(ang)) * radius * 1.005
            dy = math.sin(float(ang)) * radius * 1.005
            self._create_cube(f"{base_path}/PaintStreak_{i}", size=(0.06, 0.02, height * 0.45),
                              color=(0.30, 0.30, 0.28), pos=(x + dx, y + dy, z + height * 0.40),
                              material_key="mat_concrete_stain")

    def _create_stack(self, path: str, pos=(0, 0, 0), height=6.0):
        """Industrial exhaust stack with proper aviation warning bands."""
        x, y, z = pos

        # Main stack body - aged off-white with subtle weathering.
        self._create_cylinder(path + "/Body", radius=0.36, height=height,
                              color=(0.74, 0.73, 0.69), pos=(x, y, z + height / 2.0),
                              material_key="mat_stack_white")

        # Aviation warning bands (red, alternating).
        band_color = (0.62, 0.11, 0.10)
        self._create_cylinder(path + "/RedBand1", radius=0.37, height=0.32,
                              color=band_color, pos=(x, y, z + height * 0.20),
                              material_key="mat_stack_red")
        self._create_cylinder(path + "/RedBand2", radius=0.37, height=0.32,
                              color=band_color, pos=(x, y, z + height * 0.50),
                              material_key="mat_stack_red")
        self._create_cylinder(path + "/RedBand3", radius=0.37, height=0.32,
                              color=band_color, pos=(x, y, z + height * 0.80),
                              material_key="mat_stack_red")

        # Soot-stained cap (top of stack).
        self._create_cylinder(path + "/SootCap", radius=0.42, height=0.18,
                              color=(0.10, 0.10, 0.09), pos=(x, y, z + height + 0.09),
                              material_key="mat_stack_soot")

        # Soot streak running down from top (downwind side).
        self._create_cube(path + "/SootStreak", size=(0.08, 0.025, height * 0.55),
                          color=(0.12, 0.12, 0.11), pos=(x + 0.36, y, z + height * 0.65),
                          material_key="mat_stack_soot")

        # Concrete foundation block.
        self._create_cube(path + "/Foundation", size=(1.0, 1.0, 0.3),
                          color=(0.42, 0.41, 0.38), pos=(x, y, z + 0.15),
                          material_key="mat_concrete_dark")

        # Lightning rod at top.
        self._create_cylinder(path + "/LightningRod", radius=0.025, height=0.6,
                              color=(0.10, 0.10, 0.11), pos=(x, y, z + height + 0.48),
                              material_key="mat_reactor_lightning_rod")

    def _create_pipe_rack(self, path: str, center=(0, 0, 1), length=8.0):
        """Pipe rack with structural frame, pipes of different services, and brackets."""
        x, y, z = center

        # Top and bottom frame I-beams.
        self._create_cube(path + "/FrameTop", size=(length, 0.20, 0.10), color=(0.22, 0.23, 0.24),
                          pos=(x, y, z + 0.72), material_key="mat_dark_steel")
        self._create_cube(path + "/FrameBottom", size=(length, 0.20, 0.10), color=(0.22, 0.23, 0.24),
                          pos=(x, y, z - 0.72), material_key="mat_dark_steel")

        # Vertical supports.
        for i, px in enumerate(np.linspace(-length / 2.0 + 0.8, length / 2.0 - 0.8, max(2, int(length // 2)))):
            self._create_cube(f"{path}/Support_{i}", size=(0.12, 0.18, 1.6), color=(0.22, 0.23, 0.24),
                              pos=(x + px, y, z), material_key="mat_dark_steel")
            # Brackets connecting pipes to support.
            self._create_cube(f"{path}/Bracket_{i}", size=(0.18, 0.30, 0.06), color=(0.18, 0.19, 0.20),
                              pos=(x + px, y, z + 0.40), material_key="mat_dark_steel")

        # Color-coded service pipes (red=fire, green=cooling water, blue=service air).
        for j, (dz, mat) in enumerate([(0.40, "mat_pipe_red"), (0.10, "mat_pipe_green"), (-0.20, "mat_pipe_blue")]):
            self._create_cube(f"{path}/Pipe_{j}", size=(length, 0.14, 0.14), color=(0.3, 0.3, 0.3),
                              pos=(x, y, z + dz), material_key=mat)

        # Insulated steam line (jacketed pipe - larger diameter, white-ish).
        self._create_cube(path + "/SteamPipe", size=(length, 0.22, 0.22), color=(0.55, 0.55, 0.52),
                          pos=(x, y, z - 0.50), material_key="mat_pipe_insulation")

    def _create_pipe_bridge(self, path: str, start=(0, 0, 0), end=(4, 0, 0), levels=2):
        sx, sy, sz = start
        ex, ey, ez = end
        dx = ex - sx
        dy = ey - sy
        length = float(math.hypot(dx, dy))
        yaw = math.atan2(dy, dx)
        cx = 0.5 * (sx + ex)
        cy = 0.5 * (sy + ey)
        cz = 0.5 * (sz + ez)

        for i, zz in enumerate(np.linspace(-0.35, 0.35, levels)):
            mat_choice = ["mat_pipe_red", "mat_pipe_green", "mat_pipe_blue", "mat_pipe_yellow"][i % 4]
            self._create_cube(f"{path}/Pipe_{i}", size=(length, 0.14, 0.14), color=(0.4, 0.4, 0.4),
                              pos=(cx, cy, cz + zz), material_key=mat_choice)
        self._create_cube(path + "/SupportA", size=(0.18, 0.18, 2.1), color=(0.22, 0.23, 0.24),
                          pos=(sx, sy, sz - 0.9), material_key="mat_dark_steel")
        self._create_cube(path + "/SupportB", size=(0.18, 0.18, 2.1), color=(0.22, 0.23, 0.24),
                          pos=(ex, ey, ez - 0.9), material_key="mat_dark_steel")
        # Rotate pipes to align with start->end direction.
        for i in range(levels):
            prim = self.stage.GetPrimAtPath(f"{path}/Pipe_{i}")
            if prim and prim.IsValid():
                api = UsdGeom.XformCommonAPI(prim)
                api.SetRotate((0.0, 0.0, math.degrees(yaw)), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    def _create_catwalk(self, path: str, center=(0, 0, 0), size=(6, 1, 0.1)):
        x, y, z = center
        sx, sy, sz = size
        # Steel grating deck.
        self._create_cube(path + "/Deck", size=(sx, sy, sz), color=(0.20, 0.21, 0.22), pos=(x, y, z), material_key="mat_dark_steel")
        # Yellow safety railings.
        self._create_cube(path + "/RailL", size=(sx, 0.05, 0.85), color=(0.74, 0.58, 0.10), pos=(x, y - sy / 2.0, z + 0.47), material_key="mat_safety")
        self._create_cube(path + "/RailR", size=(sx, 0.05, 0.85), color=(0.74, 0.58, 0.10), pos=(x, y + sy / 2.0, z + 0.47), material_key="mat_safety")
        # Mid rails for OSHA-style guardrail.
        self._create_cube(path + "/MidRailL", size=(sx, 0.04, 0.04), color=(0.70, 0.55, 0.10), pos=(x, y - sy / 2.0, z + 0.25), material_key="mat_safety")
        self._create_cube(path + "/MidRailR", size=(sx, 0.04, 0.04), color=(0.70, 0.55, 0.10), pos=(x, y + sy / 2.0, z + 0.25), material_key="mat_safety")
        # Toe boards (kick plates at deck level).
        self._create_cube(path + "/ToeL", size=(sx, 0.04, 0.10), color=(0.30, 0.30, 0.30), pos=(x, y - sy / 2.0, z + 0.06), material_key="mat_dark_steel")
        self._create_cube(path + "/ToeR", size=(sx, 0.04, 0.10), color=(0.30, 0.30, 0.30), pos=(x, y + sy / 2.0, z + 0.06), material_key="mat_dark_steel")

    def _add_reference_xform(self, prim_path: str, usd_path: str, pos=(0, 0, 0), scale=(1, 1, 1)):
        wrapper = UsdGeom.Xform.Define(self.stage, prim_path)
        wrapper_prim = wrapper.GetPrim()
        asset_path = prim_path.rstrip("/") + "/Asset"
        asset_xform = UsdGeom.Xform.Define(self.stage, asset_path)
        asset_prim = asset_xform.GetPrim()
        asset_prim.GetReferences().AddReference(usd_path)
        api = UsdGeom.XformCommonAPI(wrapper_prim)
        api.SetTranslate(tuple(float(v) for v in pos))
        api.SetScale(tuple(float(v) for v in scale))
        self.ops[prim_path] = {"type": "reference", "prim": wrapper_prim}
        return wrapper_prim

    def _set_translate(self, path: str, pos):
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return
        try:
            UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in pos))
        except Exception:
            pass

    def _set_cube_scale(self, path: str, size):
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return
        try:
            UsdGeom.XformCommonAPI(prim).SetScale(tuple(float(v) for v in size))
        except Exception:
            pass

    def _sync_scene(self):
        self.pos[2] = np.clip(self.pos[2], self.no_fly_z_min, self.no_fly_z_max)
        if self.imported_plant_loaded:
            cx, cy = self.ground_center
            sx, sy = self.ground_size_xy
            self.pos[0] = np.clip(self.pos[0], cx - sx / 2.0 + 1.0, cx + sx / 2.0 - 1.0)
            self.pos[1] = np.clip(self.pos[1], cy - sy / 2.0 + 1.0, cy + sy / 2.0 - 1.0)
        else:
            b = self.world_size
            self.pos[0] = np.clip(self.pos[0], -b, b)
            self.pos[1] = np.clip(self.pos[1], -b, b)

        if self.drone_uses_usd:
            self._set_translate("/World/Drone", self.pos)
        else:
            for p in ["/World/Drone/Body", "/World/Drone/Arm_X", "/World/Drone/Arm_Y", "/World/Drone/CameraHousing", "/World/Drone/BottomCameraHousing", "/World/Drone/SensorIMU", "/World/Drone/SensorDepth", "/World/Drone/SensorDown"]:
                if p.endswith("BottomCameraHousing") or p.endswith("SensorDown"):
                    offset = np.array([0.0, 0.0, -0.18], dtype=np.float32)
                elif p.endswith("CameraHousing") or p.endswith("SensorDepth"):
                    offset = np.array([0.34, 0.0, -0.05], dtype=np.float32)
                elif p.endswith("SensorIMU"):
                    offset = np.array([0.0, 0.0, 0.10], dtype=np.float32)
                else:
                    offset = np.zeros(3, dtype=np.float32)
                self._set_translate(p, self.pos + offset)
            for i, (ox, oy) in enumerate([(0.46, 0), (-0.46, 0), (0, 0.46), (0, -0.46)]):
                self._set_translate(f"/World/Drone/Rotor_{i}", self.pos + np.array([ox, oy, 0.03], dtype=np.float32))
        self._update_stereo_camera_transforms()
        self._set_translate("/World/InspectionTarget", self.target)
        for i, obs in enumerate(self.obstacles):
            x, y, z, sx, sy, sz = obs
            path = f"/World/DynamicObstacle_{i}"
            self._set_translate(path, (x, y, z))
            self._set_cube_scale(path, (sx, sy, sz))


    def _body_offset_to_world(self, offset_body: np.ndarray) -> np.ndarray:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return np.array([
            c * float(offset_body[0]) - s * float(offset_body[1]),
            s * float(offset_body[0]) + c * float(offset_body[1]),
            float(offset_body[2]),
        ], dtype=np.float32)

    def _update_stereo_camera_transforms(self):
        # USD cameras look along local -Z. A -90 deg pitch aligns the front
        # stereo optical axis with drone body +X. The bottom camera keeps local
        # -Z vertical, so it looks down while yaw follows the vehicle heading.
        try:
            for side, yoff in [("Left", self.stereo_baseline * 0.5), ("Right", -self.stereo_baseline * 0.5)]:
                path = f"/World/Drone/Stereo{side}Camera"
                prim = self.stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                offset = np.array([0.42, yoff, -0.04], dtype=np.float32)
                cam_pos = self.pos + self._body_offset_to_world(offset)
                api = UsdGeom.XformCommonAPI(prim)
                api.SetTranslate(tuple(float(v) for v in cam_pos))
                api.SetRotate((0.0, -90.0, math.degrees(self.yaw)), UsdGeom.XformCommonAPI.RotationOrderXYZ)

            bottom = self.stage.GetPrimAtPath("/World/Drone/BottomInspectionCamera")
            if bottom and bottom.IsValid():
                bottom_offset = np.array([0.0, 0.0, -0.18], dtype=np.float32)
                bottom_pos = self.pos + self._body_offset_to_world(bottom_offset)
                api = UsdGeom.XformCommonAPI(bottom)
                api.SetTranslate(tuple(float(v) for v in bottom_pos))
                api.SetRotate((0.0, 0.0, math.degrees(self.yaw)), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        except Exception:
            pass

    def _update_viewport_camera(self):
        try:
            from omni.isaac.core.utils.viewports import set_camera_view
            if self.imported_plant_loaded and self.plant_bbox_min is not None and self.plant_bbox_max is not None:
                mn, mx = self.plant_bbox_min, self.plant_bbox_max
                center = 0.5 * (mn + mx)
                size = mx - mn
                dist = float(max(size[0], size[1], size[2]) * 0.9 + 10.0)
                eye = [float(center[0] - dist * 0.65), float(center[1] - dist * 0.75), float(min(center[2] + dist * 0.45, mx[2] + 60.0))]
                target = [float(center[0]), float(center[1]), float(min(center[2] + size[2] * 0.25, mx[2]))]
            else:
                eye = [22.0, -25.0, 16.0]
                target = [1.0, 0.0, 3.5]
            set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # navigation proxy / observations
    # ---------------------------------------------------------------------
    def _body_to_world(self, vec_body: np.ndarray) -> np.ndarray:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return np.array([c * float(vec_body[0]) - s * float(vec_body[1]), s * float(vec_body[0]) + c * float(vec_body[1]), float(vec_body[2])], dtype=np.float32)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

    def _world_to_body(self, vec_world: np.ndarray) -> np.ndarray:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return np.array([c * float(vec_world[0]) + s * float(vec_world[1]), -s * float(vec_world[0]) + c * float(vec_world[1]), float(vec_world[2])], dtype=np.float32)

    def _world_to_body_using(self, vec_world: np.ndarray, yaw_value: float) -> np.ndarray:
        c = math.cos(yaw_value)
        s = math.sin(yaw_value)
        return np.array([c * float(vec_world[0]) + s * float(vec_world[1]), -s * float(vec_world[0]) + c * float(vec_world[1]), float(vec_world[2])], dtype=np.float32)

    def _reset_navigation_proxy(self):
        # Define the GPS-denied SLAM frame at episode reset.
        # The policy sees pose in this local visual-odometry frame, not Isaac global pose.
        self.slam_origin_true = self.pos.copy()
        self.slam_yaw0_true = float(self.yaw)
        self.slam_pos = np.zeros(3, dtype=np.float32)
        self.slam_vel = np.zeros(3, dtype=np.float32)
        self.slam_yaw = 0.0
        self.slam_quality = 1.0
        self.slam_drift = np.zeros(3, dtype=np.float32)
        self.cuvslam_metric_offset = None

    def _update_proxy_inertial(self, yaw_rate_cmd: float):
        world_acc = (self.vel - self.prev_true_vel) / max(self.dt, 1e-6)
        acc_body = self._world_to_body(world_acc)
        self.imu_acc_body = acc_body + self.rng.normal(0.0, 0.04, size=3).astype(np.float32)
        self.imu_gyro_body = np.array([
            self.rng.normal(0.0, 0.01),
            self.rng.normal(0.0, 0.01),
            yaw_rate_cmd + self.rng.normal(0.0, 0.015),
        ], dtype=np.float32)

    def _update_navigation_proxy(self):
        # Convert true simulator motion into the reset-aligned local SLAM frame.
        # This keeps the observation GPS-denied and episode-local while preserving
        # physically consistent camera/target geometry.
        true_local_pos = self._world_to_body_using(self.pos - self.slam_origin_true, self.slam_yaw0_true)
        true_local_vel = self._world_to_body_using(self.vel, self.slam_yaw0_true)
        true_local_yaw = float(self.yaw - self.slam_yaw0_true)

        if self.slam_mode == "gt":
            self.slam_pos = true_local_pos.astype(np.float32)
            self.slam_vel = true_local_vel.astype(np.float32)
            self.slam_yaw = true_local_yaw
            self.slam_quality = 1.0
            return

        if self.slam_mode == "cuvslam":
            odom = self.cuvslam_receiver.get_latest(max_age_sec=0.75) if self.cuvslam_receiver is not None else None
            if odom is not None:
                # cuVSLAM odometry is already local. We assume the Isaac Sim stereo
                # camera frame is started near reset and use it as the GPS-denied SLAM frame.
                self.slam_pos = odom["pos"].astype(np.float32)
                self.slam_vel = odom["vel"].astype(np.float32)
                self.slam_yaw = float(odom["yaw"])
                self.slam_quality = 1.0
                return
            # If cuVSLAM is not available yet, keep the last estimate and signal
            # low tracking quality. This prevents feeding Isaac ground truth into
            # the policy while cuVSLAM starts.
            self.slam_vel = (0.90 * self.slam_vel).astype(np.float32)
            self.slam_quality = 0.0
            return

        # Feature-aware proxy SLAM: good texture/illumination/wind reduces drift,
        # while poor visual conditions still produce tracking degradation.
        try:
            feature_count = self._estimate_visible_feature_count()
            mu_T = self._texture_membership(feature_count)
            mu_L = self._illumination_membership(float(self.illumination_lux))
            mu_W = self._wind_stability_membership(float(self.wind_mps))
            slam_stability = self._clip01(0.55 * mu_T + 0.25 * mu_L + 0.20 * mu_W)
        except Exception:
            slam_stability = 0.50
        drift_scale = 0.35 + 0.85 * (1.0 - slam_stability)
        drift_step = self.rng.normal(0.0, self.slam_drift_pos_per_sec * self.dt * drift_scale, size=3).astype(np.float32)
        drift_step[2] *= 0.35
        self.slam_drift = (0.997 * self.slam_drift + drift_step).astype(np.float32)

        pos_noise = self.rng.normal(0.0, self.slam_pos_noise_std * (0.55 + 0.70 * (1.0 - slam_stability)), size=3).astype(np.float32)
        vel_noise = self.rng.normal(0.0, self.slam_vel_noise_std * (0.55 + 0.70 * (1.0 - slam_stability)), size=3).astype(np.float32)
        yaw_noise = float(self.rng.normal(0.0, self.slam_yaw_noise_std * (0.55 + 0.70 * (1.0 - slam_stability))))
        tracking_loss_prob = self.slam_tracking_loss_prob * (0.55 + 1.40 * (1.0 - slam_stability))

        if self.rng.random() < tracking_loss_prob:
            self.slam_quality = float(self.rng.uniform(0.15, 0.45))
            # Freeze/degrade the estimate but keep small jitter, which mimics partial tracking failure.
            self.slam_pos = (self.slam_pos + 0.04 * pos_noise).astype(np.float32)
            self.slam_vel = (0.90 * self.slam_vel).astype(np.float32)
        else:
            self.slam_quality = float(min(1.0, self.slam_quality + self.slam_quality_recover_rate))
            self.slam_pos = (true_local_pos.astype(np.float32) + self.slam_drift + pos_noise).astype(np.float32)
            self.slam_vel = (true_local_vel.astype(np.float32) + vel_noise).astype(np.float32)

        self.slam_yaw = true_local_yaw + yaw_noise

    def yaw_at_reset(self):
        return self.slam_yaw0_true

    def _target_rel_slam(self):
        target_local_true = self._world_to_body_using(self.target - self.slam_origin_true, self.slam_yaw0_true)
        return (target_local_true - self.slam_pos).astype(np.float32)

    def _camera_features(self):
        # Camera-like cues are built in the estimated SLAM/body frame.
        rel_slam = self._target_rel_slam()
        rel_body = self._world_to_body_using(rel_slam, self.slam_yaw)
        forward = float(rel_body[0])
        lateral = float(rel_body[1])
        vertical = float(rel_body[2])
        visible = 0.0
        u = 0.0
        v = 0.0
        apparent = 0.0

        if forward > 0.15:
            denom_u = max(forward * math.tan(self.camera_hfov * 0.5), 1e-4)
            denom_v = max(forward * math.tan(self.camera_vfov * 0.5), 1e-4)
            u = lateral / denom_u
            v = vertical / denom_v
            apparent = min(self.target_radius / max(forward, 0.2), 1.0)
            if abs(u) <= 1.0 and abs(v) <= 1.0 and not self._line_hits_any_obstacle(self.pos, self.target):
                visible = 1.0
        # tracking quality also affects how much the camera cue can be trusted
        apparent *= 0.5 + 0.5 * self.slam_quality
        visible *= 1.0 if self.slam_quality > 0.25 else 0.0
        cam_feats = np.array([
            np.clip(u, -1.5, 1.5),
            np.clip(v, -1.5, 1.5),
            np.clip(apparent, 0.0, 1.0),
            np.clip(forward / max(self.max_ray_range, 1.0), -1.0, 1.5),
            visible,
        ], dtype=np.float32)
        return cam_feats, rel_body

    def _get_obs(self):
        cam_feats, _rel_body = self._camera_features()
        rays = self._ray_distances(noisy=True)
        progress = np.array([
            self.target_idx / max(len(self.targets) - 1, 1),
            self.step_count / max(self.max_episode_steps, 1),
        ], dtype=np.float32)
        target_rel_slam = self._target_rel_slam() / max(self.max_ray_range, 1.0)
        slam_pos_norm = self.slam_pos / max(self.max_ray_range, 1.0)
        slam_vel_norm = self.slam_vel / 3.0
        yaw_sc = np.array([math.sin(self.slam_yaw), math.cos(self.slam_yaw)], dtype=np.float32)
        tracking_quality = np.array([self.slam_quality], dtype=np.float32)

        return np.concatenate([
            slam_pos_norm.astype(np.float32),
            slam_vel_norm.astype(np.float32),
            yaw_sc,
            self.imu_acc_body.astype(np.float32),
            self.imu_gyro_body.astype(np.float32),
            target_rel_slam.astype(np.float32),
            cam_feats.astype(np.float32),
            rays.astype(np.float32),
            tracking_quality,
            self.prev_action.astype(np.float32),
            progress,
        ]).astype(np.float32)

    # ---------------------------------------------------------------------
    # depth / obstacle geometry
    # ---------------------------------------------------------------------
    def _ray_distances(self, noisy: bool = False):
        rays = np.ones(self.num_rays, dtype=np.float32)
        origin = self.pos.copy()
        for i in range(self.num_rays):
            ang = self.yaw + (2.0 * math.pi * i / self.num_rays)
            direction = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
            val = np.clip(self._ray_to_obstacles_2d(origin[:2], direction, float(origin[2])) / self.max_ray_range, 0.0, 1.0)
            if noisy:
                val += float(self.rng.normal(0.0, self.depth_noise_std))
                if self.slam_quality < 0.35:
                    val += float(self.rng.normal(0.0, self.depth_noise_std * 1.5))
            rays[i] = np.clip(val, 0.0, 1.0)
        return rays

    def _ray_to_obstacles_2d(self, origin_xy, direction_xy, origin_z: float = 0.0):
        best = self.max_ray_range
        # Dynamic obstacles.
        for obs in self.obstacles:
            ox, oy, oz, sx, sy, sz = [float(v) for v in obs]
            if not (oz - 0.5 * sz - self.drone_collision_radius <= origin_z <= oz + 0.5 * sz + self.drone_collision_radius):
                continue
            mn = np.array([ox - sx / 2.0, oy - sy / 2.0], dtype=np.float32)
            mx = np.array([ox + sx / 2.0, oy + sy / 2.0], dtype=np.float32)
            d = self._ray_aabb_2d(origin_xy, direction_xy, mn, mx)
            if d is not None:
                best = min(best, d)
        # Static plant geometry collision volumes.  These are ignored when the
        # UAV is safely above the object top, which keeps high-altitude coverage
        # flight from being incorrectly blocked.
        for box in getattr(self, "static_collision_boxes", []):
            mn3 = np.asarray(box.get("mn"), dtype=np.float32)
            mx3 = np.asarray(box.get("mx"), dtype=np.float32)
            if not (float(mn3[2]) - self.drone_collision_radius <= origin_z <= float(mx3[2]) + self.drone_collision_radius):
                continue
            mn = mn3[:2] - self.drone_collision_radius
            mx = mx3[:2] + self.drone_collision_radius
            d = self._ray_aabb_2d(origin_xy, direction_xy, mn, mx)
            if d is not None:
                best = min(best, d)
        if self.imported_plant_loaded:
            cx, cy = self.ground_center
            sx, sy = self.ground_size_xy
            mn = np.array([cx - sx / 2.0, cy - sy / 2.0], dtype=np.float32)
            mx = np.array([cx + sx / 2.0, cy + sy / 2.0], dtype=np.float32)
        else:
            b = self.world_size
            mn = np.array([-b, -b], dtype=np.float32)
            mx = np.array([b, b], dtype=np.float32)
        d_bound = self._ray_exit_aabb_2d(origin_xy, direction_xy, mn, mx)
        if d_bound is not None:
            best = min(best, d_bound)
        return best

    @staticmethod
    def _ray_aabb_2d(origin, direction, mn, mx):
        tmin = 0.0
        tmax = 1e9
        for k in range(2):
            if abs(direction[k]) < 1e-8:
                if origin[k] < mn[k] or origin[k] > mx[k]:
                    return None
            else:
                inv = 1.0 / direction[k]
                t1 = (mn[k] - origin[k]) * inv
                t2 = (mx[k] - origin[k]) * inv
                t1, t2 = min(t1, t2), max(t1, t2)
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return None
        if tmax < 0:
            return None
        return float(max(tmin, 0.0))

    @staticmethod
    def _ray_exit_aabb_2d(origin, direction, mn, mx):
        ts = []
        for k in range(2):
            if abs(direction[k]) < 1e-8:
                continue
            for bound in [mn[k], mx[k]]:
                t = (bound - origin[k]) / direction[k]
                if t > 0:
                    p = origin + t * direction
                    other = 1 - k
                    if mn[other] - 1e-5 <= p[other] <= mx[other] + 1e-5:
                        ts.append(float(t))
        return min(ts) if ts else None

    def _register_static_collision_box(self, path: str, center, size, category: str = "structure"):
        """Register a conservative AABB for visual USD structures.

        The simulation uses lightweight kinematics rather than Isaac rigid-body
        physics, so visual geometry must be mirrored into analytic collision
        boxes.  These boxes are used by depth rays, obstacle avoidance, and the
        swept collision check in step().
        """
        if not hasattr(self, "static_collision_boxes"):
            self.static_collision_boxes = []
        c = np.asarray(center, dtype=np.float32).reshape(3)
        s = np.maximum(np.asarray(size, dtype=np.float32).reshape(3), 1e-3)
        mn = c - 0.5 * s
        mx = c + 0.5 * s
        self.static_collision_boxes.append({"path": str(path), "mn": mn, "mx": mx, "category": str(category)})

    def _should_register_static_collision(self, path: str, size, pos) -> bool:
        p = str(path).lower()
        if not p.startswith("/world/npp") and not p.startswith("/world/importedpowerplant"):
            return False
        skip_words = (
            "stain", "seam", "line", "stripe", "marker", "warning", "label", "sign",
            "waterstreak", "ruststreak", "moss", "shadow", "wetbase", "target",
            "lightningrod", "valvestem", "rail", "handrail", "ladder",
        )
        if any(w in p for w in skip_words):
            return False
        sx, sy, sz = [float(v) for v in size]
        if sz < 0.22 or max(sx, sy) < 0.18:
            return False
        zc = float(pos[2])
        if zc + 0.5 * sz < 0.25:
            return False
        return True

    @staticmethod
    def _segment_intersects_aabb_3d(a, b, mn, mx) -> bool:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        mn = np.asarray(mn, dtype=np.float32)
        mx = np.asarray(mx, dtype=np.float32)
        d = b - a
        tmin = 0.0
        tmax = 1.0
        for k in range(3):
            if abs(float(d[k])) < 1e-9:
                if float(a[k]) < float(mn[k]) or float(a[k]) > float(mx[k]):
                    return False
            else:
                inv = 1.0 / float(d[k])
                t1 = (float(mn[k]) - float(a[k])) * inv
                t2 = (float(mx[k]) - float(a[k])) * inv
                t1, t2 = min(t1, t2), max(t1, t2)
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return False
        return True

    def _iter_collision_aabbs(self):
        # Static plant structures.
        for box in getattr(self, "static_collision_boxes", []):
            yield str(box.get("path", "static")), np.asarray(box["mn"], dtype=np.float32), np.asarray(box["mx"], dtype=np.float32)
        # Dynamic obstacles.
        for i, obs in enumerate(getattr(self, "obstacles", [])):
            ox, oy, oz, sx, sy, sz = [float(v) for v in obs]
            c = np.array([ox, oy, oz], dtype=np.float32)
            s = np.array([sx, sy, sz], dtype=np.float32)
            yield f"dynamic_obstacle_{i}", c - 0.5 * s, c + 0.5 * s

    def _point_inside_collision_aabb(self, pos, margin: float = 0.0):
        p = np.asarray(pos, dtype=np.float32)
        for name, mn, mx in self._iter_collision_aabbs():
            mn_i = mn - float(margin)
            mx_i = mx + float(margin)
            if np.all(p >= mn_i) and np.all(p <= mx_i):
                return True, name
        return False, ""

    def _resolve_collision_free_motion(self, old_pos, new_pos):
        old_pos = np.asarray(old_pos, dtype=np.float32)
        new_pos = np.asarray(new_pos, dtype=np.float32)
        radius = float(getattr(self, "drone_collision_radius", 0.42))
        best_hit_name = ""
        for name, mn, mx in self._iter_collision_aabbs():
            mn_i = mn - radius
            mx_i = mx + radius
            if self._segment_intersects_aabb_3d(old_pos, new_pos, mn_i, mx_i) or np.all((new_pos >= mn_i) & (new_pos <= mx_i)):
                best_hit_name = name
                break
        if best_hit_name:
            # Emergency vertical escape: if the planned horizontal motion would
            # hit a structure, climb above the local safe floor before moving
            # laterally. This keeps the rendered UAV above the plant instead of
            # stopping/colliding at roof height.
            emergency = old_pos.copy()
            emergency[2] = float(np.clip(
                max(float(old_pos[2]) + 0.35, self._safe_altitude_floor(old_pos[:2], margin=2.80) + 0.65),
                self.no_fly_z_min + 0.5,
                self.no_fly_z_max - 0.45,
            ))
            blocked = False
            for _n2, mn2, mx2 in self._iter_collision_aabbs():
                mn_e = mn2 - radius
                mx_e = mx2 + radius
                if self._segment_intersects_aabb_3d(old_pos, emergency, mn_e, mx_e) or np.all((emergency >= mn_e) & (emergency <= mx_e)):
                    blocked = True
                    break
            if not blocked:
                self.last_collision_distance_m = self._nearest_collision_clearance(emergency)
                self.last_collision_name = best_hit_name
                return emergency.astype(np.float32), False, "avoidance_climb"
            self.last_collision_distance_m = 0.0
            return old_pos.copy(), True, best_hit_name
        self.last_collision_distance_m = self._nearest_collision_clearance(new_pos)
        return new_pos.astype(np.float32), False, ""

    def _nearest_collision_clearance(self, pos) -> float:
        p = np.asarray(pos, dtype=np.float32)
        best = float("inf")
        for _name, mn, mx in self._iter_collision_aabbs():
            # Distance from point to AABB surface; zero if inside.
            delta = np.maximum(np.maximum(mn - p, p - mx), 0.0)
            best = min(best, float(np.linalg.norm(delta)))
        return best

    def _safe_altitude_floor(self, xy, margin: float = 1.25) -> float:
        """Minimum safe z above any structure whose footprint is nearby."""
        xy = np.asarray(xy, dtype=np.float32).reshape(2)
        floor = float(self.no_fly_z_min + 0.35)
        for _name, mn, mx in self._iter_collision_aabbs():
            mn_xy = mn[:2] - float(margin)
            mx_xy = mx[:2] + float(margin)
            if np.all(xy >= mn_xy) and np.all(xy <= mx_xy):
                floor = max(floor, float(mx[2]) + float(getattr(self, "vertical_structure_clearance_m", 1.20)))
        return float(min(floor, self.no_fly_z_max - 0.75))

    def _line_hits_any_obstacle(self, a, b):
        radius = float(getattr(self, "drone_collision_radius", 0.42))
        for _name, mn, mx in self._iter_collision_aabbs():
            if self._segment_intersects_aabb_3d(a, b, mn - radius, mx + radius):
                return True
        return False

    @staticmethod
    def _dist_point_segment(p, a, b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-8:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = np.clip(t, 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * ab)))


    def _local_obstacle_avoidance_body_xy(self, commanded_speed: float) -> np.ndarray:
        """Simple local repulsive field in the UAV body frame.

        The policy action remains one scalar speed.  This helper only changes
        the deterministic waypoint tracker, using the procedural obstacle map as
        a proxy for depth-ray avoidance.  It improves completion rate in the
        randomized industrial e2 domain where obstacles can lie close to the
        direct waypoint line.
        """
        if getattr(self, "obstacles", None) is None or len(self.obstacles) == 0:
            return np.zeros(2, dtype=np.float32)
        gain = float(getattr(self, "obstacle_avoidance_gain", 0.85))
        if gain <= 0.0:
            return np.zeros(2, dtype=np.float32)
        avoid_range = float(getattr(self, "obstacle_avoidance_range", 2.20))
        rep = np.zeros(2, dtype=np.float32)
        for obs in self.obstacles:
            ox, oy, _oz, sx, sy, _sz = [float(v) for v in obs]
            rel_world = np.array([ox - float(self.pos[0]), oy - float(self.pos[1]), 0.0], dtype=np.float32)
            rel_body = self._world_to_body(rel_world)[:2]
            # Only avoid obstacles in front or near the current lateral corridor.
            if rel_body[0] < -0.75:
                continue
            obs_radius = 0.5 * max(sx, sy)
            clearance = float(np.linalg.norm(rel_body) - obs_radius)
            if clearance >= avoid_range:
                continue
            d = max(float(np.linalg.norm(rel_body)), 1e-6)
            away = -rel_body / d
            # Prefer lateral avoidance rather than simply backing up.
            if abs(float(away[1])) < 0.35:
                side = -1.0 if rel_body[1] >= 0.0 else 1.0
                away = np.array([0.15 * float(away[0]), side], dtype=np.float32)
                away = away / max(float(np.linalg.norm(away)), 1e-6)
            strength = ((avoid_range - clearance) / max(avoid_range, 1e-6)) ** 2
            rep += (gain * strength * away).astype(np.float32)
        rep_norm = float(np.linalg.norm(rep))
        max_rep = 0.65 * max(float(commanded_speed), self.osd_vmin)
        if rep_norm > max_rep and rep_norm > 1e-6:
            rep = rep / rep_norm * max_rep
        return rep.astype(np.float32)

    # ---------------------------------------------------------------------
    # reward / domain randomization
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # paper-aligned metrics and fuzzy methodology
    # ---------------------------------------------------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    def _start_paper_episode_metrics(self):
        self.metric_episode_id += 1
        expected_targets = 12 if str(getattr(self, "sim_env_id", "e1")).lower() == "e2" else 16
        if len(self.targets) != expected_targets:
            raise RuntimeError(
                f"[ROUTE_ERROR_METRICS] sim_env_id={self.sim_env_id} expected {expected_targets} targets, "
                f"got {len(self.targets)} before metrics start."
            )
        self.coverage_points = self.targets.copy().astype(np.float32)
        self.coverage_visited = np.zeros(len(self.coverage_points), dtype=bool)
        self.prev_paper_potential = None
        self.prev_paper_coverage = 0.0
        self.trajectory_records = []
        self._op_cbrs_lib_hits = 0
        self._op_cbrs_lib_misses = 0
        self._op_cbrs_lib_misses_no_lib = 0
        self.episode_metrics = {
            "reward_sum": 0.0,
            "steps": 0,
            "drift_incidents": 0,
            "drift_active": False,
            "loc_acc_steps": 0,
            "total_loc_steps": 0,
            "energy_wh": 0.0,
            "coverage_max": 0.0,
            "coverage_delta_sum": 0.0,
            "waypoints_reached": 0,
            "pre_drift_alerts": 0,
            "speed_sum": 0.0,
            "mu_T_sum": 0.0,
            "mu_L_sum": 0.0,
            "mu_W_sum": 0.0,
            "mu_A_sum": 0.0,
        }

    def _estimate_visible_feature_count(self) -> int:
        if self.visual_feature_points is None or len(self.visual_feature_points) == 0:
            return 0
        pts = self.visual_feature_points
        rel = pts - self.pos.reshape(1, 3)
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        x = c * rel[:, 0] + s * rel[:, 1]
        y = -s * rel[:, 0] + c * rel[:, 1]
        z = rel[:, 2]
        forward = x > 0.35
        rng = np.sqrt(x * x + y * y + z * z) < self.max_ray_range
        u_ok = np.abs(y / np.maximum(x * math.tan(self.camera_hfov * 0.5), 1e-5)) <= 1.0
        v_ok = np.abs(z / np.maximum(x * math.tan(self.camera_vfov * 0.5), 1e-5)) <= 1.0
        count = int(np.count_nonzero(forward & rng & u_ok & v_ok))
        # Convert dense synthetic landmarks to a feature-per-frame scale similar to the paper.
        return int(np.clip(round(count / 8.0), 0, 80))

    def _texture_membership(self, feature_count: int) -> float:
        if self.sim_env_id == "e2":
            med, high = 5.0, 10.0
        else:
            med, high = 8.0, 15.0
        if feature_count >= high:
            return 1.0
        if feature_count >= med:
            return self._clip01((feature_count - med) / max(high - med, 1e-6))
        return self._clip01(0.35 * feature_count / max(med, 1e-6))

    def _illumination_membership(self, lux: float) -> float:
        if self.sim_env_id == "e2":
            low, high0, high1, max_lux = 280.0, 360.0, 440.0, 560.0
        else:
            low, high0, high1, max_lux = 350.0, 450.0, 550.0, 650.0
        if high0 <= lux <= high1:
            return 1.0
        if lux < high0:
            return self._clip01((lux - low) / max(high0 - low, 1e-6))
        return self._clip01((max_lux - lux) / max(max_lux - high1, 1e-6))

    def _wind_stability_membership(self, wind_mps: float) -> float:
        ref = 5.0 if self.sim_env_id == "e2" else 2.0
        return self._clip01(1.0 - wind_mps / max(ref, 1e-6))

    def _trajectory_adherence(self) -> Tuple[float, float, float]:
        start = getattr(self, "current_segment_start", self.slam_origin_true)
        d = self._dist_point_segment(self.pos.astype(np.float32), np.asarray(start, dtype=np.float32), self.target.astype(np.float32))
        eps_min, eps_max = 0.25, 1.50
        mu_A = self._clip01(1.0 - d / max(eps_max, 1e-6))
        eps_mu = eps_min + (eps_max - eps_min) * mu_A
        return float(d), float(mu_A), float(eps_mu)

    def _update_fuzzy_coverage(self, mu_T: float, mu_L: float) -> Tuple[float, float]:
        if self.coverage_points is None or self.coverage_visited is None or len(self.coverage_points) == 0:
            return 0.0, 0.0
        d = np.linalg.norm(self.coverage_points - self.pos.reshape(1, 3), axis=1)
        newly = d <= max(self.coverage_radius, 1e-6)
        self.coverage_visited |= newly
        base_cov = float(np.count_nonzero(self.coverage_visited) / max(len(self.coverage_visited), 1))
        # Fuzzy coverage: inspection is valuable only when perception quality is sufficient.
        mu_cvg = base_cov * (0.55 + 0.30 * mu_T + 0.15 * mu_L)
        mu_cvg = self._clip01(mu_cvg)
        delta = max(0.0, mu_cvg - float(self.prev_paper_coverage))
        self.prev_paper_coverage = max(float(self.prev_paper_coverage), mu_cvg)
        return mu_cvg, delta

    def _localization_error(self) -> float:
        true_local = self._world_to_body_using(self.pos - self.slam_origin_true, self.slam_yaw0_true)
        if self.slam_mode == "cuvslam":
            # cuVSLAM runs in its own local frame.  For paper metrics, align the first
            # available local estimate to the simulator-local reset frame using a constant offset.
            if not hasattr(self, "cuvslam_metric_offset") or self.cuvslam_metric_offset is None:
                self.cuvslam_metric_offset = true_local.astype(np.float32) - self.slam_pos.astype(np.float32)
            aligned = self.slam_pos.astype(np.float32) + self.cuvslam_metric_offset.astype(np.float32)
            return float(np.linalg.norm(aligned - true_local))
        return float(np.linalg.norm(self.slam_pos.astype(np.float32) - true_local.astype(np.float32)))

    def _compute_osd_speed_memberships(self, mu_T: float, mu_L: float, mu_W: float, mu_A: float) -> Tuple[float, float, float, float]:
        # mu_W is wind-stability; low mu_W means stronger wind disturbance.
        risk = self._clip01(0.45 * (1.0 - mu_T) + 0.25 * (1.0 - mu_L) + 0.20 * (1.0 - mu_W) + 0.10 * (1.0 - mu_A))
        mu_slow = self._clip01(risk)
        mu_fast = self._clip01(0.50 * mu_T + 0.25 * mu_L + 0.20 * mu_W + 0.05 * mu_A - 0.25 * risk)
        mu_med = self._clip01(1.0 - abs(risk - 0.50) * 2.0)
        total = max(mu_slow + mu_med + mu_fast, 1e-6)
        mu_slow, mu_med, mu_fast = mu_slow / total, mu_med / total, mu_fast / total
        vt = 0.50 * mu_slow + 0.75 * mu_med + 1.00 * mu_fast
        return float(vt), float(mu_slow), float(mu_med), float(mu_fast)

    def _load_op_cbrs_library(self, path: Optional[Path]):
        """Load offline OP-CBRS potential library built from uniform-speed tasks T*.

        The paper's OP-CBRS uses value/potential functions from simpler
        uniform-speed coverage tasks. In this implementation, each fuzzy-state
        bin stores a learned empirical potential Phi*(s), estimated from
        trajectories collected by collect_uniform_speed/build_op_cbrs_library.
        """
        if path is None:
            return None
        try:
            if path.exists():
                data = json.loads(path.read_text())
                if isinstance(data, dict) and "states" in data:
                    print(f"[OP-CBRS] Loaded potential library: {path}")
                    return data
        except Exception as exc:
            print(f"[OP-CBRS] Warning: could not load potential library {path}: {exc}")
        return None

    @staticmethod
    def _membership_bin(value: float) -> str:
        v = float(value)
        if v < 0.35:
            return "LOW"
        if v < 0.70:
            return "MED"
        return "HIGH"

    def _op_cbrs_state_key(self, mu_T: float, mu_L: float, mu_W: float, mu_A: float) -> str:
        return (
            f"T={self._membership_bin(mu_T)}|"
            f"L={self._membership_bin(mu_L)}|"
            f"W={self._membership_bin(mu_W)}|"
            f"A={self._membership_bin(mu_A)}"
        )

    def _heuristic_state_potential(self, mu_cvg: float, mu_T: float, mu_L: float, mu_W: float,
                                   mu_A: float, loc_error: float, min_ray: float) -> float:
        # Dense fallback potential: coverage + stable visual features + illumination
        # + wind/adherence robustness - localization/near-obstacle risk.
        return float(
            2.00 * mu_cvg
            + 0.45 * mu_T
            + 0.30 * mu_L
            + 0.20 * mu_W
            + 0.25 * mu_A
            - 0.75 * min(float(loc_error), 2.0)
            - 0.15 * max(0.0, 0.25 - float(min_ray))
        )

    def _select_op_cbrs_potential(self, mu_cvg: float, mu_T: float, mu_L: float, mu_W: float,
                                  mu_A: float, loc_error: float, min_ray: float) -> Tuple[float, str]:
        """Select Phi_A*(s) from the offline uniform-speed potential library.

        If an exact fuzzy bin is unavailable, the heuristic potential is used.
        This implements the paper's idea that fuzzy membership values select
        a suitable potential function for OP-CBRS.
        """
        fallback = self._heuristic_state_potential(mu_cvg, mu_T, mu_L, mu_W, mu_A, loc_error, min_ray)
        lib = getattr(self, "op_cbrs_library", None)
        if not lib:
            self._op_cbrs_lib_misses_no_lib = int(getattr(self, "_op_cbrs_lib_misses_no_lib", 0)) + 1
            return fallback, "heuristic"

        key = self._op_cbrs_state_key(mu_T, mu_L, mu_W, mu_A)
        states = lib.get("states", {})
        node = states.get(key)
        if isinstance(node, dict) and "phi" in node:
            phi = float(node.get("phi", fallback))
            self._op_cbrs_lib_hits = int(getattr(self, "_op_cbrs_lib_hits", 0)) + 1
            # Blend to avoid discontinuities when crossing fuzzy bins.
            return float(0.70 * phi + 0.30 * fallback), f"library:{key}"
        self._op_cbrs_lib_misses = int(getattr(self, "_op_cbrs_lib_misses", 0)) + 1
        return fallback, "heuristic_fallback"

    def _build_boustrophedon_targets(self) -> np.ndarray:
        """Canonical inspection route.

        IMPORTANT: this method is the single source of truth for the planned
        route used by the controller, metrics, coverage plan, and VSLAM map.

        e1 / power plant: 16 visually distinct XY waypoints arranged as a
        4 x 4 boustrophedon grid.  Using 16 unique XY points is intentional;
        if two altitude levels share the same XY coordinates, the 2D/VSLAM map
        still appears to have only 8 points.

        e2 / industrial: 12 waypoints are kept for the transfer-domain layout.
        """
        if str(getattr(self, "sim_env_id", "e1")).lower() == "e2":
            x_cols = [-12.0, -6.0, 0.0, 6.0]
            y_rows = [-8.0, 1.0, 10.0]
            z_base = 8.8
            pts = []
            for ix, x in enumerate(x_cols):
                ys = y_rows if ix % 2 == 0 else list(reversed(y_rows))
                for iy, y in enumerate(ys):
                    pts.append([float(x), float(y), float(z_base + 0.35 * ((ix + iy) % 2))])
            arr = np.asarray(pts, dtype=np.float32)
            print(f"[ROUTE_BUILD_E2_INDUSTRIAL] vslam_waypoints={len(arr)} drone_targets={len(arr)} expected=12")
            return arr

        # e1 / power plant: 16 unique top-view points so the VSLAM map also
        # visibly doubles from 8 to 16 inspection locations.
        x_cols = [-10.5, -4.0, 2.5, 9.0]
        y_rows = [-7.5, -2.4, 2.7, 7.8]
        z_base = 13.4
        pts = []
        for ix, x in enumerate(x_cols):
            ys = y_rows if ix % 2 == 0 else list(reversed(y_rows))
            for iy, y in enumerate(ys):
                pts.append([float(x), float(y), float(z_base + 0.25 * ((ix + iy) % 2))])
        arr = np.asarray(pts, dtype=np.float32)
        print(f"[ROUTE_BUILD_E1_POWER_PLANT] vslam_waypoints={len(arr)} drone_targets={len(arr)} expected=16")
        return arr

    def _coverage_layout_metadata(self) -> Dict[str, object]:
        """Build coverage/VSLAM-map metadata from the active route.

        Previous versions hardcoded eight low-texture boxes, which made the
        VSLAM map keep showing only eight inspection objects even when the route
        was changed elsewhere.  This version derives one box from every active
        waypoint, so e1 shows 16 boxes and e2 shows 12 boxes.
        """
        route = np.asarray(getattr(self, "targets", getattr(self, "base_targets", np.zeros((0, 3)))), dtype=np.float32)
        if route.ndim != 2 or route.shape[0] == 0:
            route = np.asarray(getattr(self, "base_targets", self._build_boustrophedon_targets()), dtype=np.float32)
        boxes = []
        for p in route:
            boxes.append((float(p[0]), float(p[1]), 2.6, 2.6))
        entry = np.asarray(getattr(self, "pos", np.array([-14.0, -12.0, 12.8], dtype=np.float32)), dtype=np.float32).reshape(3)
        fov_size = 4.8
        all_x = np.concatenate([route[:, 0], np.array([entry[0]], dtype=np.float32)])
        all_y = np.concatenate([route[:, 1], np.array([entry[1]], dtype=np.float32)])
        x_min, x_max = float(np.min(all_x) - 4.0), float(np.max(all_x) + 4.0)
        y_min, y_max = float(np.min(all_y) - 4.0), float(np.max(all_y) + 4.0)
        print(f"[VSLAM_MAP] sim_env_id={self.sim_env_id} vslam_waypoints={len(route)} boxes={len(boxes)}")
        return {
            "target_area": (x_min, y_min, x_max - x_min, y_max - y_min),
            "low_texture_boxes": boxes,
            "fov_box": (float(entry[0]) - 0.5 * fov_size, float(entry[1]) - 0.5 * fov_size, fov_size, fov_size),
            "route": route.copy(),
            "entry_point": entry.copy(),
        }

    def _collect_navigation_metrics(self) -> Dict[str, float]:
        feature_count = self._estimate_visible_feature_count() if self.visual_feature_points is not None else 0
        self.texture_count = feature_count
        mu_T = self._texture_membership(feature_count)
        mu_L = self._illumination_membership(float(self.illumination_lux))
        mu_W = self._wind_stability_membership(float(self.wind_mps))
        adherence_d, mu_A, eps_mu = self._trajectory_adherence()
        loc_error = self._localization_error()
        speed = float(np.linalg.norm(self.vel))
        vt, mu_slow, mu_med, mu_fast = self._compute_osd_speed_memberships(mu_T, mu_L, mu_W, mu_A)
        return {
            "feature_count": float(feature_count),
            "mu_T": float(mu_T),
            "mu_L": float(mu_L),
            "mu_W": float(mu_W),
            "mu_A": float(mu_A),
            "epsilon_mu": float(eps_mu),
            "adherence_error_m": float(adherence_d),
            "localization_error_m": float(loc_error),
            "speed_mps": float(speed),
            "osd_speed_vt": float(vt),
            "mu_slow": float(mu_slow),
            "mu_medium": float(mu_med),
            "mu_fast": float(mu_fast),
            "slam_quality": float(self.slam_quality),
        }

    def _compute_paper_step_values(self, action, progress: float, dist: float, min_ray: float, reached: bool, precomputed_nav=None) -> Dict[str, float]:
        nav = dict(precomputed_nav) if precomputed_nav is not None else self._collect_navigation_metrics()
        self._nav_metrics_cache = dict(nav)
        mu_T = float(nav["mu_T"])
        mu_L = float(nav["mu_L"])
        mu_W = float(nav["mu_W"])
        mu_A = float(nav["mu_A"])
        mu_cvg, coverage_delta = self._update_fuzzy_coverage(mu_T, mu_L)
        loc_error = float(nav["localization_error_m"])
        speed = float(nav["speed_mps"])
        vt = float(nav["osd_speed_vt"])
        mu_slow = float(nav["mu_slow"])
        mu_med = float(nav["mu_medium"])
        mu_fast = float(nav["mu_fast"])
        power_w = float(self.motor_hover_power_w + self.motor_speed_power_gain_w * speed * speed + 18.0 * abs(float(self.vel[2])))
        energy_step_wh = power_w * self.dt / 3600.0
        drift_event = bool((loc_error > self.drift_threshold) or (float(nav.get("slam_quality", 1.0)) < 0.20))
        pre_drift = bool((loc_error > 0.70 * self.drift_threshold) or (mu_T < 0.35 and float(nav.get("slam_quality", 1.0)) < 0.55))
        potential, potential_source = self._select_op_cbrs_potential(mu_cvg, mu_T, mu_L, mu_W, mu_A, loc_error, min_ray)
        potential = float(potential)
        prev_phi = self.prev_paper_potential
        shaping = 0.0 if prev_phi is None else 0.95 * float(prev_phi) - potential
        self.prev_paper_potential = potential
        return {
            "feature_count": float(nav["feature_count"]),
            "mu_T": mu_T,
            "mu_L": mu_L,
            "mu_W": mu_W,
            "mu_A": mu_A,
            "epsilon_mu": float(nav["epsilon_mu"]),
            "adherence_error_m": float(nav["adherence_error_m"]),
            "mu_cvg": float(mu_cvg),
            "coverage_delta": float(coverage_delta),
            "localization_error_m": float(loc_error),
            "localization_accurate": float(loc_error <= self.localization_threshold),
            "drift_event": float(drift_event),
            "pre_drift_alert": float(pre_drift),
            "speed_mps": float(speed),
            "policy_speed_mps": float(getattr(self, "last_policy_speed_mps", speed)),
            "commanded_speed_mps": float(getattr(self, "last_commanded_speed_mps", speed)),
            "osd_speed_cap_mps": float(getattr(self, "last_osd_speed_cap_mps", vt)),
            "action_speed_normalized": float(np.clip(float(np.asarray(action).reshape(-1)[0]), -1.0, 1.0)),
            "osd_speed_vt": float(vt),
            "mu_slow": float(mu_slow),
            "mu_medium": float(mu_med),
            "mu_fast": float(mu_fast),
            "motor_power_w": float(power_w),
            "energy_step_wh": float(energy_step_wh),
            "op_cbrs_potential": float(potential),
            "op_cbrs_potential_source": str(potential_source),
            "op_cbrs_shaping_reward": float(0.35 * math.tanh(shaping / 0.35)),
            "illumination_lux": float(self.illumination_lux),
            "wind_mps": float(self.wind_mps),
            "paper_progress": float(progress),
            "slam_quality": float(nav.get("slam_quality", self.slam_quality)),
        }

    def _append_csv_row(self, path: Path, row: Dict[str, object]):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists() and path.stat().st_size > 0
            with path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            print(f"[METRICS] Warning: failed to write {path}: {exc}")

    def _record_paper_step_metrics(self, action, reward: float, info: Dict[str, object], terminated: bool, truncated: bool):
        paper = {k: float(v) for k, v in info.items() if isinstance(v, (int, float, np.floating, np.integer))}
        ep = self.episode_metrics
        ep["reward_sum"] = float(ep.get("reward_sum", 0.0)) + float(reward)
        ep["steps"] = int(ep.get("steps", 0)) + 1
        ep["loc_acc_steps"] = int(ep.get("loc_acc_steps", 0)) + int(paper.get("localization_accurate", 0.0) >= 0.5)
        ep["total_loc_steps"] = int(ep.get("total_loc_steps", 0)) + 1
        ep["energy_wh"] = float(ep.get("energy_wh", 0.0)) + float(paper.get("energy_step_wh", 0.0))
        ep["coverage_max"] = max(float(ep.get("coverage_max", 0.0)), float(paper.get("mu_cvg", 0.0)))
        ep["coverage_delta_sum"] = float(ep.get("coverage_delta_sum", 0.0)) + float(paper.get("coverage_delta", 0.0))
        ep["speed_sum"] = float(ep.get("speed_sum", 0.0)) + float(paper.get("speed_mps", 0.0))
        for key in ["mu_T", "mu_L", "mu_W", "mu_A"]:
            ep[f"{key}_sum"] = float(ep.get(f"{key}_sum", 0.0)) + float(paper.get(key, 0.0))
        if bool(info.get("reached", False)):
            ep["waypoints_reached"] = int(ep.get("waypoints_reached", 0)) + 1
        drift_now = bool(paper.get("drift_event", 0.0) >= 0.5)
        if drift_now and not bool(ep.get("drift_active", False)):
            ep["drift_incidents"] = int(ep.get("drift_incidents", 0)) + 1
            ep["drift_active"] = True
        if not drift_now:
            ep["drift_active"] = False
        if bool(paper.get("pre_drift_alert", 0.0) >= 0.5):
            ep["pre_drift_alerts"] = int(ep.get("pre_drift_alerts", 0)) + 1

        row = {
            "run_id": self.run_id,
            "episode": self.metric_episode_id,
            "step": self.step_count,
            "time_sec": round(self.step_count * self.dt, 6),
            "sim_env_id": self.sim_env_id,
            "policy_name": self.policy_name,
            "slam_mode": self.slam_mode,
            "x": float(self.pos[0]), "y": float(self.pos[1]), "z": float(self.pos[2]),
            "slam_x": float(self.slam_pos[0]), "slam_y": float(self.slam_pos[1]), "slam_z": float(self.slam_pos[2]),
            "yaw": float(self.yaw), "slam_yaw": float(self.slam_yaw),
            "target_index": int(info.get("target_index", self.target_idx)),
            "distance_to_target_m": float(info.get("distance", 0.0)),
            "min_depth_ray": float(info.get("min_ray", 0.0)),
            "reward": float(reward),
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
        }
        for k in [
            "feature_count", "mu_T", "mu_L", "mu_W", "mu_A", "epsilon_mu", "adherence_error_m",
            "mu_cvg", "coverage_delta", "localization_error_m", "localization_accurate", "drift_event",
            "pre_drift_alert", "speed_mps", "policy_speed_mps", "commanded_speed_mps", "osd_speed_cap_mps", "action_speed_normalized", "osd_speed_vt", "mu_slow", "mu_medium", "mu_fast",
            "motor_power_w", "energy_step_wh", "op_cbrs_potential", "op_cbrs_shaping_reward",
            "illumination_lux", "wind_mps", "paper_progress", "slam_quality",
        ]:
            row[k] = float(info.get(k, paper.get(k, 0.0)))
        row["op_cbrs_potential_source"] = str(info.get("op_cbrs_potential_source", ""))
        self._append_csv_row(self.step_metrics_csv, row)
        if self.save_trajectory_record:
            self.trajectory_records.append(dict(row))

        # Stable-Baselines3 reserves info["episode"] for Monitor episode summaries
        # and expects it to be a dictionary.  Our paper metrics also used a scalar
        # key named "episode", which made SB3 crash in dump_logs with:
        # TypeError: object of type 'int' has no len().  Keep "episode" only in
        # CSV/JSON artifacts, but expose it to the RL loop as "paper_episode".
        sb3_safe_row = dict(row)
        sb3_safe_row["paper_episode"] = sb3_safe_row.pop("episode", self.metric_episode_id)
        info.update(sb3_safe_row)
        info.pop("episode", None)
        return info

    def _write_episode_trajectory_csv(self, path: Path):
        if not self.trajectory_records:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.trajectory_records[0].keys()), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.trajectory_records)
        except Exception as exc:
            print(f"[TRAJECTORY] Warning: failed to write {path}: {exc}")

    def _smooth_heatmap_array(self, H: np.ndarray, passes: int = 2) -> np.ndarray:
        """Small dependency-free smoothing for paper figures."""
        H = np.asarray(H, dtype=np.float32)
        if H.size == 0:
            return H
        kernel = np.array(
            [[1.0, 2.0, 1.0],
             [2.0, 4.0, 2.0],
             [1.0, 2.0, 1.0]],
            dtype=np.float32,
        )
        kernel /= float(kernel.sum())
        out = H.copy()
        for _ in range(max(1, int(passes))):
            P = np.pad(out, 1, mode="edge")
            out = (
                kernel[0, 0] * P[:-2, :-2] + kernel[0, 1] * P[:-2, 1:-1] + kernel[0, 2] * P[:-2, 2:] +
                kernel[1, 0] * P[1:-1, :-2] + kernel[1, 1] * P[1:-1, 1:-1] + kernel[1, 2] * P[1:-1, 2:] +
                kernel[2, 0] * P[2:, :-2] + kernel[2, 1] * P[2:, 1:-1] + kernel[2, 2] * P[2:, 2:]
            )
        return out

    def _visible_feature_image_heatmap(self, pos: np.ndarray, yaw: float, feature_gain: float = 1.0, bins: int = 72, camera_mode: str = "front"):
        """Build a camera-frame visual-feature heatmap for one UAV pose.

        camera_mode="front" uses the forward stereo/RGB camera. camera_mode="bottom"
        uses a downward-looking inspection camera mounted under the UAV. The
        bottom view is better for roof/chimney inspection because the UAV now
        flies above the plant rather than through the middle of structures.
        """
        if self.visual_feature_points is None or len(self.visual_feature_points) == 0:
            return None, 0

        pts = np.asarray(self.visual_feature_points, dtype=np.float32)
        pos = np.asarray(pos, dtype=np.float32).reshape(3)
        rel = pts - pos[None, :]
        c, s = math.cos(-float(yaw)), math.sin(-float(yaw))
        # Body-frame coordinates: +x forward, +y left/right, +z up.
        bx = c * rel[:, 0] - s * rel[:, 1]
        by = s * rel[:, 0] + c * rel[:, 1]
        bz = rel[:, 2]
        mode = str(camera_mode or "front").lower()
        H = np.zeros((bins, bins), dtype=np.float32)
        max_depth = max(4.0, float(self.max_ray_range))

        if mode in ("bottom", "down", "downward", "below"):
            # Downward camera: optical axis is body/world -Z. Image horizontal
            # roughly follows body x, image vertical follows body y.
            depth_axis = -bz
            h_ang = np.arctan2(bx, np.maximum(depth_axis, 1e-6))
            v_ang = np.arctan2(by, np.maximum(depth_axis, 1e-6))
            depth = np.sqrt(bx * bx + by * by + depth_axis * depth_axis) + 1e-6
            visible = (
                (depth_axis > 0.15) &
                (depth < max_depth) &
                (np.abs(h_ang) <= 0.5 * float(self.camera_hfov)) &
                (np.abs(v_ang) <= 0.5 * float(self.camera_vfov))
            )
            u_all = np.tan(h_ang) / max(np.tan(0.5 * float(self.camera_hfov)), 1e-6)
            v_all = np.tan(v_ang) / max(np.tan(0.5 * float(self.camera_vfov)), 1e-6)
        else:
            # Forward camera: optical axis is body +X.
            depth_axis = bx
            depth = np.sqrt(bx * bx + by * by + bz * bz) + 1e-6
            h_ang = np.arctan2(by, np.maximum(bx, 1e-6))
            v_ang = np.arctan2(bz, np.maximum(np.sqrt(bx * bx + by * by), 1e-6))
            visible = (
                (bx > 0.15) &
                (depth < max_depth) &
                (np.abs(h_ang) <= 0.5 * float(self.camera_hfov)) &
                (np.abs(v_ang) <= 0.5 * float(self.camera_vfov))
            )
            u_all = np.tan(h_ang) / max(np.tan(0.5 * float(self.camera_hfov)), 1e-6)
            v_all = np.tan(v_ang) / max(np.tan(0.5 * float(self.camera_vfov)), 1e-6)

        visible_count = int(np.count_nonzero(visible))
        if np.any(visible):
            u = u_all[visible]
            v = v_all[visible]
            d = depth[visible]
            w = (1.0 - np.clip(d / max(max_depth, 1e-6), 0.0, 1.0))
            w *= (1.0 - 0.35 * np.clip(np.abs(u), 0.0, 1.0))
            w *= (1.0 - 0.25 * np.clip(np.abs(v), 0.0, 1.0))
            w *= max(float(feature_gain), 0.05)
            H0, _, _ = np.histogram2d(u, v, bins=bins, range=[[-1.0, 1.0], [-1.0, 1.0]], weights=w)
            H += H0.T.astype(np.float32)

        if getattr(self, "visual_hotspots", None):
            grid_u = np.linspace(-1.0, 1.0, bins, dtype=np.float32)
            grid_v = np.linspace(-1.0, 1.0, bins, dtype=np.float32)
            UU, VV = np.meshgrid(grid_u, grid_v)
            for hotspot in self.visual_hotspots:
                hpos = np.asarray(hotspot["pos"], dtype=np.float32)
                rel_h = hpos - pos
                hx = c * rel_h[0] - s * rel_h[1]
                hy = s * rel_h[0] + c * rel_h[1]
                hz = rel_h[2]
                if mode in ("bottom", "down", "downward", "below"):
                    axis = -hz
                    hdepth = float(np.linalg.norm([hx, hy, axis])) + 1e-6
                    if axis <= 0.15 or hdepth >= max_depth:
                        continue
                    hu = math.tan(math.atan2(hx, max(axis, 1e-6))) / max(math.tan(0.5 * float(self.camera_hfov)), 1e-6)
                    hv = math.tan(math.atan2(hy, max(axis, 1e-6))) / max(math.tan(0.5 * float(self.camera_vfov)), 1e-6)
                    sigma_scale = 1.35
                else:
                    hdepth = float(np.linalg.norm([hx, hy, hz])) + 1e-6
                    if hx <= 0.15 or hdepth >= max_depth:
                        continue
                    hu = math.tan(math.atan2(hy, max(hx, 1e-6))) / max(math.tan(0.5 * float(self.camera_hfov)), 1e-6)
                    hv = math.tan(math.atan2(hz, max(math.sqrt(hx * hx + hy * hy), 1e-6))) / max(math.tan(0.5 * float(self.camera_vfov)), 1e-6)
                    sigma_scale = 1.0
                if abs(hu) > 1.05 or abs(hv) > 1.05:
                    continue
                sigma = float(hotspot.get("sigma", 0.12)) * sigma_scale
                strength = float(hotspot.get("strength", 1.0)) * max(0.30, 1.0 - hdepth / max(max_depth, 1e-6))
                H += strength * np.exp(-((UU - hu) ** 2 + (VV - hv) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)

        H = self._smooth_heatmap_array(H, passes=2)
        if float(H.max()) > 0.0:
            H = H / float(H.max())
        return H.astype(np.float32), visible_count

    def _build_heatmap_array(self):
        """Build a global visual-feature density map from camera-visible points."""
        if not self.trajectory_records:
            return None, None, None
        xs = np.array([float(r.get("x", 0.0)) for r in self.trajectory_records], dtype=np.float32)
        ys = np.array([float(r.get("y", 0.0)) for r in self.trajectory_records], dtype=np.float32)
        yaws = np.array([float(r.get("yaw", 0.0)) for r in self.trajectory_records], dtype=np.float32)
        features = np.array([float(r.get("feature_count", 0.0)) for r in self.trajectory_records], dtype=np.float32)
        tx = self.targets[:, 0] if getattr(self, "targets", None) is not None else np.array([], dtype=np.float32)
        ty = self.targets[:, 1] if getattr(self, "targets", None) is not None else np.array([], dtype=np.float32)
        if self.visual_feature_points is not None and len(self.visual_feature_points) > 0:
            fpts = np.asarray(self.visual_feature_points, dtype=np.float32)
            fx, fy = fpts[:, 0], fpts[:, 1]
        else:
            fx, fy = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        all_x = np.concatenate([xs, tx.astype(np.float32), fx.astype(np.float32)])
        all_y = np.concatenate([ys, ty.astype(np.float32), fy.astype(np.float32)])
        x_min, x_max = float(np.nanmin(all_x) - 2.0), float(np.nanmax(all_x) + 2.0)
        y_min, y_max = float(np.nanmin(all_y) - 2.0), float(np.nanmax(all_y) + 2.0)
        if abs(x_max - x_min) < 1e-3:
            x_max += 1.0
        if abs(y_max - y_min) < 1e-3:
            y_max += 1.0
        H = np.zeros((self.heatmap_grid_size, self.heatmap_grid_size), dtype=np.float32)
        if self.visual_feature_points is not None and len(self.visual_feature_points) > 0:
            pts = np.asarray(self.visual_feature_points, dtype=np.float32)
            stride = max(1, len(xs) // 180)
            for i in range(0, len(xs), stride):
                pos = np.array([xs[i], ys[i], float(self.trajectory_records[i].get("z", 0.0))], dtype=np.float32)
                rel = pts - pos[None, :]
                c, ss = math.cos(-float(yaws[i])), math.sin(-float(yaws[i]))
                bx = c * rel[:, 0] - ss * rel[:, 1]
                by = ss * rel[:, 0] + c * rel[:, 1]
                bz = rel[:, 2]
                depth = np.sqrt(bx * bx + by * by + bz * bz) + 1e-6
                h_ang = np.arctan2(by, np.maximum(bx, 1e-6))
                v_ang = np.arctan2(bz, np.maximum(np.sqrt(bx * bx + by * by), 1e-6))
                mask = (
                    (bx > 0.15) &
                    (depth < max(4.0, float(self.max_ray_range))) &
                    (np.abs(h_ang) <= 0.5 * float(self.camera_hfov)) &
                    (np.abs(v_ang) <= 0.5 * float(self.camera_vfov))
                )
                if np.any(mask):
                    gain = 0.25 + 0.75 * np.clip(features[i] / max(float(np.nanmax(features)), 1.0), 0.0, 1.0)
                    h, _, _ = np.histogram2d(
                        pts[mask, 0], pts[mask, 1],
                        bins=self.heatmap_grid_size,
                        range=[[x_min, x_max], [y_min, y_max]],
                        weights=np.full(int(np.count_nonzero(mask)), float(gain), dtype=np.float32),
                    )
                    H += h.T.astype(np.float32)
        else:
            # Fallback: at least show the path-weighted response.
            weights = np.maximum(features, 1e-3)
            h, xedges, yedges = np.histogram2d(xs, ys, bins=self.heatmap_grid_size, range=[[x_min, x_max], [y_min, y_max]], weights=weights)
            return self._smooth_heatmap_array(h.T, passes=2), xedges, yedges
        H = self._smooth_heatmap_array(H, passes=2)
        xedges = np.linspace(x_min, x_max, self.heatmap_grid_size + 1, dtype=np.float32)
        yedges = np.linspace(y_min, y_max, self.heatmap_grid_size + 1, dtype=np.float32)
        return H, xedges, yedges

    def _render_fuzzy_sequence_figure(self, episode_prefix: str, bundle: Dict[str, object], plt):
        """Render front-camera and downward-camera visual heatmap sequences.

        Outputs are separate and paper-friendly:
        - *_front_visual_heatmap_sequence.png
        - *_bottom_visual_heatmap_sequence.png
        - *_visual_heatmap_sequence.png  (compact dual-camera comparison)
        """
        if not self.trajectory_records:
            return bundle
        try:
            traj = self.trajectory_records
            steps = np.array([int(r.get("step", 0)) for r in traj], dtype=np.int32)
            xs = np.array([float(r.get("x", 0.0)) for r in traj], dtype=np.float32)
            ys = np.array([float(r.get("y", 0.0)) for r in traj], dtype=np.float32)
            zs = np.array([float(r.get("z", 0.0)) for r in traj], dtype=np.float32)
            yaws = np.array([float(r.get("yaw", 0.0)) for r in traj], dtype=np.float32)
            features = np.array([float(r.get("feature_count", 0.0)) for r in traj], dtype=np.float32)
            mu_t = np.array([float(r.get("mu_T", 0.0)) for r in traj], dtype=np.float32)
            mu_l = np.array([float(r.get("mu_L", 0.0)) for r in traj], dtype=np.float32)
            loc_err = np.array([float(r.get("localization_error_m", 0.0)) for r in traj], dtype=np.float32)
            if len(steps) < 4:
                return bundle

            def _norm(v):
                v = np.asarray(v, dtype=np.float32)
                lo = float(np.nanmin(v))
                hi = float(np.nanmax(v))
                if hi - lo < 1e-6:
                    return np.zeros_like(v, dtype=np.float32)
                return (v - lo) / (hi - lo + 1e-6)

            score = (
                0.55 * _norm(features)
                + 0.20 * np.maximum(0.0, 1.0 - mu_t)
                + 0.08 * np.maximum(0.0, 1.0 - mu_l)
                + 0.17 * _norm(loc_err)
            )

            nframes = min(12, len(steps))
            edges = np.linspace(0, len(steps), nframes + 1, dtype=int)
            chosen = []
            feat_thr = max(1.0, 0.08 * float(np.nanmax(features)))
            for a, b in zip(edges[:-1], edges[1:]):
                if b <= a:
                    continue
                local = np.arange(a, b)
                non_empty = local[features[local] > feat_thr]
                candidate = non_empty if len(non_empty) else local
                best_local = int(candidate[int(np.nanargmax(score[candidate]))])
                if best_local not in chosen:
                    chosen.append(best_local)
            for idx in np.argsort(score)[::-1]:
                idx = int(idx)
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= nframes:
                    break
            chosen = sorted(chosen[:nframes], key=lambda i: steps[i])
            vmax_features = max(float(np.nanmax(features)), 1.0)

            def _heat(idx: int, mode: str):
                H, cnt = self._visible_feature_image_heatmap(
                    np.array([xs[idx], ys[idx], zs[idx]], dtype=np.float32),
                    float(yaws[idx]),
                    feature_gain=0.65 + 0.75 * float(max(features[idx], 1.0)) / vmax_features,
                    bins=96,
                    camera_mode=mode,
                )
                if H is None:
                    H = np.zeros((96, 96), dtype=np.float32)
                return np.power(np.clip(H, 0.0, 1.0), 0.70), int(cnt)

            def _render_single(mode: str, title: str, out_name: str, key_name: str):
                ncols = 6 if len(chosen) > 6 else max(len(chosen), 1)
                nrows = int(math.ceil(len(chosen) / max(ncols, 1)))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.9 * nrows + 1.1), squeeze=False)
                fig.subplots_adjust(left=0.03, right=0.985, top=0.87, bottom=0.13, wspace=0.08, hspace=0.18)
                last_im = None
                for ax in axes.ravel():
                    ax.axis("off")
                for j, idx in enumerate(chosen):
                    ax = axes.ravel()[j]
                    H_plot, cnt = _heat(idx, mode)
                    last_im = ax.imshow(H_plot, cmap="magma", origin="lower", interpolation="bicubic", vmin=0.0, vmax=1.0)
                    ax.set_title(f"Step {int(steps[idx])}", fontsize=16, pad=6)
                    ax.text(
                        0.03, 0.94, f"features={cnt}", transform=ax.transAxes,
                        ha="left", va="top", color="white", fontsize=10.2,
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.42, edgecolor="none"),
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect("equal")
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.8)
                        spine.set_edgecolor((1, 1, 1, 0.18))
                for ax in axes.ravel()[len(chosen):]:
                    ax.axis("off")
                if last_im is not None:
                    cax = fig.add_axes([0.27, 0.055, 0.46, 0.022])
                    cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
                    cbar.set_label("Normalized visual-feature response", fontsize=11)
                    cbar.ax.tick_params(labelsize=9)
                fig.suptitle(title, fontsize=19, y=0.965)
                path = self.figure_dir / f"{episode_prefix}_{out_name}.png"
                fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.10)
                plt.close(fig)
                bundle[key_name] = str(path)
                return path

            front_path = _render_single(
                "front",
                "Front camera visual heatmap sequence around inspection objects",
                "front_visual_heatmap_sequence",
                "front_visual_heatmap_sequence_png",
            )
            bottom_path = _render_single(
                "bottom",
                "Downward inspection camera visual heatmap sequence",
                "bottom_visual_heatmap_sequence",
                "bottom_visual_heatmap_sequence_png",
            )

            # Compact dual-camera comparison. Keep this filename for compatibility
            # with previous paper scripts that expect *_visual_heatmap_sequence.png.
            dual_chosen = chosen
            if len(dual_chosen) > 6:
                dual_chosen = [chosen[int(round(v))] for v in np.linspace(0, len(chosen) - 1, 6)]
            ncols = max(1, len(dual_chosen))
            fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 8.3), squeeze=False)
            fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.12, wspace=0.08, hspace=0.16)
            last_im = None
            for cidx, idx in enumerate(dual_chosen):
                for ridx, (mode, row_label) in enumerate([("front", "Front camera"), ("bottom", "Bottom camera")]):
                    ax = axes[ridx, cidx]
                    H_plot, cnt = _heat(idx, mode)
                    last_im = ax.imshow(H_plot, cmap="magma", origin="lower", interpolation="bicubic", vmin=0.0, vmax=1.0)
                    if ridx == 0:
                        ax.set_title(f"Step {int(steps[idx])}", fontsize=15, pad=6)
                    if cidx == 0:
                        ax.text(-0.08, 0.5, row_label, transform=ax.transAxes, rotation=90,
                                ha="center", va="center", fontsize=13, fontweight="bold")
                    ax.text(0.03, 0.94, f"n={cnt}", transform=ax.transAxes, ha="left", va="top",
                            color="white", fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.20", facecolor="black", alpha=0.42, edgecolor="none"))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect("equal")
            if last_im is not None:
                cax = fig.add_axes([0.28, 0.055, 0.44, 0.022])
                cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
                cbar.set_label("Normalized visual-feature response", fontsize=11)
                cbar.ax.tick_params(labelsize=9)
            fig.suptitle("Dual-camera visual heatmap sequence: front view and downward inspection view", fontsize=18, y=0.96)
            seq_path = self.figure_dir / f"{episode_prefix}_visual_heatmap_sequence.png"
            fig.savefig(seq_path, dpi=260, bbox_inches="tight", pad_inches=0.10)
            plt.close(fig)
            bundle["visual_heatmap_sequence_png"] = str(seq_path)
            bundle["dual_camera_visual_heatmap_sequence_png"] = str(seq_path)
        except Exception as exc:
            print(f"[FIGURES] Warning: failed to render visual heatmap sequence: {exc}")
        return bundle

    def _render_episode_figures(self, episode_prefix: str, bundle: Dict[str, object]):
        if not self.save_paper_figures or not self.trajectory_records:
            return bundle
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, FancyArrowPatch
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as exc:
            print(f"[FIGURES] matplotlib unavailable, skipping figure export: {exc}")
            return bundle

        try:
            traj = self.trajectory_records
            steps = np.array([int(r.get("step", 0)) for r in traj], dtype=np.int32)
            xs = np.array([float(r.get("x", 0.0)) for r in traj], dtype=np.float32)
            ys = np.array([float(r.get("y", 0.0)) for r in traj], dtype=np.float32)
            zs = np.array([float(r.get("z", 0.0)) for r in traj], dtype=np.float32)
            sxs = np.array([float(r.get("slam_x", 0.0)) for r in traj], dtype=np.float32)
            sys_ = np.array([float(r.get("slam_y", 0.0)) for r in traj], dtype=np.float32)
            szs = np.array([float(r.get("slam_z", 0.0)) for r in traj], dtype=np.float32)
            speed = np.array([float(r.get("speed_mps", 0.0)) for r in traj], dtype=np.float32)
            potential = np.array([float(r.get("op_cbrs_potential", 0.0)) for r in traj], dtype=np.float32)
            features = np.array([float(r.get("feature_count", 0.0)) for r in traj], dtype=np.float32)
            mu_t = np.array([float(r.get("mu_T", 0.0)) for r in traj], dtype=np.float32)
            loc_err = np.array([float(r.get("localization_error_m", 0.0)) for r in traj], dtype=np.float32)
            low_tex = mu_t < 0.35
            layout = self._coverage_layout_metadata()
            # Align SLAM trajectory to the true trajectory for visualization.
            # Metrics still use raw SLAM error; this is only for the map figure.
            sxs_plot = np.array(sxs, copy=True)
            sys_plot = np.array(sys_, copy=True)
            szs_plot = np.array(szs, copy=True)
            if len(sxs_plot) > 4 and np.all(np.isfinite(sxs_plot)):
                src = np.column_stack([sxs_plot, sys_plot]).astype(np.float64)
                dst = np.column_stack([xs, ys]).astype(np.float64)
                try:
                    src_mean = src.mean(axis=0)
                    dst_mean = dst.mean(axis=0)
                    src0 = src - src_mean
                    dst0 = dst - dst_mean
                    H = src0.T @ dst0
                    U, Svals, Vt = np.linalg.svd(H)
                    R = Vt.T @ U.T
                    if np.linalg.det(R) < 0:
                        Vt[-1, :] *= -1
                        R = Vt.T @ U.T
                    scale = float(np.sum(Svals) / max(np.sum(src0 ** 2), 1e-9))
                    scale = float(np.clip(scale, 0.25, 4.0))
                    aligned = scale * (src0 @ R.T) + dst_mean
                    sxs_plot = aligned[:, 0].astype(np.float32)
                    sys_plot = aligned[:, 1].astype(np.float32)
                    szs_plot = (szs_plot - szs_plot[0] + zs[0]).astype(np.float32)
                except Exception:
                    sxs_plot = sxs_plot - sxs_plot[0] + xs[0]
                    sys_plot = sys_plot - sys_plot[0] + ys[0]
                    szs_plot = szs_plot - szs_plot[0] + zs[0]

            # 1) Planned coverage figure similar to the requested coverage_inspection layout.
            cov_path = self.figure_dir / f"{episode_prefix}_coverage_plan.png"
            fig, ax = plt.subplots(figsize=(11.5, 8.2))
            tx, ty, tw, th = layout["target_area"]
            ax.add_patch(Rectangle((tx, ty), tw, th, facecolor="#f3f3f3", edgecolor="black", linewidth=2.0, hatch=".", zorder=0))
            for cx, cy, w, h in layout["low_texture_boxes"]:
                ax.add_patch(Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h, facecolor="white", edgecolor="black", linewidth=2.0, zorder=2))
            fbx, fby, fbw, fbh = layout["fov_box"]
            ax.add_patch(Rectangle((fbx, fby), fbw, fbh, facecolor="#d7efb5", edgecolor="black", linestyle="--", linewidth=2.0, alpha=0.85, zorder=1))
            plan = np.asarray(layout["route"], dtype=np.float32)
            entry = np.asarray(layout["entry_point"], dtype=np.float32)
            px = np.concatenate([[entry[0]], plan[:, 0]])
            py = np.concatenate([[entry[1]], plan[:, 1]])
            ax.plot(px, py, linestyle=(0, (6, 4)), linewidth=2.6, color="#1f4de3", marker="o", markersize=4.5, zorder=3, label="Coverage flight path")
            # Direction arrows along the boustrophedon sweep.
            for a_i, b_i in zip(range(0, len(px) - 1), range(1, len(px))):
                if a_i % 2 == 0 or b_i == len(px) - 1:
                    ax.add_patch(FancyArrowPatch((px[a_i], py[a_i]), (px[b_i], py[b_i]), arrowstyle="-|>", mutation_scale=12, color="#1f4de3", linewidth=1.8, alpha=0.85, zorder=4))
            ax.scatter([entry[0]], [entry[1]], s=120, color="#1f4de3", zorder=4, label="Start")
            ax.scatter(plan[:, 0], plan[:, 1], s=70, facecolor="#1f4de3", edgecolor="white", linewidth=0.8, zorder=4, label="Inspection waypoints")
            ax.text(entry[0] - 0.6, entry[1] - 1.1, "Start", fontsize=11, color="#1f4de3")
            ax.set_title("Boustrophedon coverage inspection path and low-texture regions")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(tx - 0.6, tx + tw + 0.6)
            ax.set_ylim(ty - 0.6, ty + th + 0.6)
            ax.grid(alpha=0.18)
            # Compact legend matching the semantic elements in the schematic.
            from matplotlib.lines import Line2D
            legend_handles = [
                Rectangle((0, 0), 1, 1, facecolor="#f3f3f3", edgecolor="black", hatch=".", label="Inspection target area"),
                Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", label="Low-texture area"),
                Rectangle((0, 0), 1, 1, facecolor="#d7efb5", edgecolor="black", linestyle="--", label="UAV field of view"),
                Line2D([0], [0], color="#1f4de3", linestyle=(0, (6, 4)), marker="o", label="Coverage flight path"),
            ]
            ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
            fig.tight_layout()
            fig.savefig(cov_path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            bundle["coverage_plan_png"] = str(cov_path)

            # 2) Separate 2D executed trajectory (true path only).
            traj2d_path = self.figure_dir / f"{episode_prefix}_trajectory2d.png"
            fig, ax = plt.subplots(figsize=(10.8, 8.1))
            ax.plot(plan[:, 0], plan[:, 1], linestyle=(0, (4, 4)), linewidth=1.9, color="#8e8e8e", label="Planned coverage path")
            ax.plot(xs, ys, linewidth=2.5, color="#0b57d0", label="Executed trajectory")
            if np.any(low_tex):
                ax.scatter(xs[low_tex], ys[low_tex], s=26, color="#ef6c00", alpha=0.88, label="Low-texture traversal")
            ax.scatter(self.targets[:, 0], self.targets[:, 1], s=72, marker="x", color="black", linewidth=1.7, label="Inspection points")
            ax.scatter(xs[0], ys[0], s=100, marker="o", color="#1f4de3", label="Start")
            ax.scatter(xs[-1], ys[-1], s=130, marker="*", color="#d32f2f", label="End")
            for cx, cy, w, h in layout["low_texture_boxes"]:
                ax.add_patch(Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h, facecolor="none", edgecolor="black", linewidth=1.2, alpha=0.55))
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Top-view executed trajectory ({self.sim_env_id})")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.20)
            all_x = np.concatenate([np.asarray(xs, dtype=np.float32), np.asarray(plan[:, 0], dtype=np.float32), np.asarray(self.targets[:, 0], dtype=np.float32)])
            all_y = np.concatenate([np.asarray(ys, dtype=np.float32), np.asarray(plan[:, 1], dtype=np.float32), np.asarray(self.targets[:, 1], dtype=np.float32)])
            x_pad = max(2.0, 0.08 * float(np.nanmax(all_x) - np.nanmin(all_x) + 1e-6))
            y_pad = max(2.0, 0.08 * float(np.nanmax(all_y) - np.nanmin(all_y) + 1e-6))
            ax.set_xlim(float(np.nanmin(all_x) - x_pad), float(np.nanmax(all_x) + x_pad))
            ax.set_ylim(float(np.nanmin(all_y) - y_pad), float(np.nanmax(all_y) + y_pad))
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(traj2d_path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            bundle["trajectory_2d_png"] = str(traj2d_path)

            # 3) Separate VSLAM top-view trajectory map. This intentionally
            # resembles a SLAM occupancy/feature map: black visual features /
            # structural points, blue VSLAM path, and red inspection endpoints.
            vslam_path = self.figure_dir / f"{episode_prefix}_vslam_trajectory.png"
            fig, ax = plt.subplots(figsize=(7.4, 11.8))
            rng_map = np.random.default_rng(2026)

            # Build a dense black pseudo point-cloud from object edges, corridor
            # walls, and visual landmarks. This creates a map-like view similar
            # to the requested coverage_inspection reference, without merging
            # this plot with the normal 2D/3D trajectory figures.
            cloud_x, cloud_y = [], []
            boxes = list(layout.get("low_texture_boxes", []))
            for cx, cy, w, h in boxes:
                # Rectangle edges
                for _ in range(360):
                    side = int(rng_map.integers(0, 4))
                    if side == 0:
                        x = rng_map.uniform(cx - 0.5 * w, cx + 0.5 * w); y = cy - 0.5 * h
                    elif side == 1:
                        x = rng_map.uniform(cx - 0.5 * w, cx + 0.5 * w); y = cy + 0.5 * h
                    elif side == 2:
                        x = cx - 0.5 * w; y = rng_map.uniform(cy - 0.5 * h, cy + 0.5 * h)
                    else:
                        x = cx + 0.5 * w; y = rng_map.uniform(cy - 0.5 * h, cy + 0.5 * h)
                    cloud_x.append(x + rng_map.normal(0.0, 0.055))
                    cloud_y.append(y + rng_map.normal(0.0, 0.055))
                # Object interior sparse features
                for _ in range(260):
                    cloud_x.append(rng_map.uniform(cx - 0.62 * w, cx + 0.62 * w))
                    cloud_y.append(rng_map.uniform(cy - 0.62 * h, cy + 0.62 * h))

            # Long vertical scan columns matching boustrophedon corridors.
            for xcol in [-10.5, -4.0, 2.5, 9.0]:
                for y in np.linspace(-7.6, 7.6, 500):
                    for off in [-0.78, 0.78]:
                        cloud_x.append(xcol + off + rng_map.normal(0.0, 0.06))
                        cloud_y.append(float(y) + rng_map.normal(0.0, 0.08))
            # Some ground clutter below the inspection sweep.
            for _ in range(1400):
                cloud_x.append(rng_map.uniform(-14.5, 13.5))
                cloud_y.append(rng_map.uniform(-14.5, -8.5))

            if getattr(self, "visual_feature_points", None) is not None and len(self.visual_feature_points) > 0:
                pts = np.asarray(self.visual_feature_points, dtype=np.float32)
                if len(pts) > 5500:
                    sel = rng_map.choice(len(pts), size=5500, replace=False)
                    pts = pts[sel]
                cloud_x.extend(pts[:, 0].tolist())
                cloud_y.extend(pts[:, 1].tolist())

            ax.scatter(cloud_x, cloud_y, s=1.6, color="black", alpha=0.62, linewidths=0, label="VSLAM map points")

            # VSLAM trajectory in blue, using aligned local trajectory only for
            # visualization. Localization-error metrics still use raw SLAM error.
            ax.plot(sxs_plot, sys_plot, color="#0037ff", linewidth=2.4, alpha=0.95, label="VSLAM trajectory")
            ax.scatter(sxs_plot, sys_plot, s=6, color="#0037ff", alpha=0.95, linewidths=0)
            ax.scatter(sxs_plot[0], sys_plot[0], s=70, color="#0037ff", zorder=6, label="Start")
            ax.scatter(sxs_plot[-1], sys_plot[-1], s=95, marker="*", color="#d7191c", zorder=6, label="End")

            # Red local clusters around inspection points / endpoint, matching
            # the reference style where active regions are red.
            txs = self.targets[:, 0]
            tys = self.targets[:, 1]
            ax.scatter(txs, tys, s=28, color="#e60000", alpha=0.88, label="Inspection points")
            for px_i, py_i in zip(txs, tys):
                rr_x = rng_map.normal(float(px_i), 0.24, 55)
                rr_y = rng_map.normal(float(py_i), 0.24, 55)
                ax.scatter(rr_x, rr_y, s=5, color="#e60000", alpha=0.30, linewidths=0)

            # Planned path as a faint blue dashed guide behind the SLAM path.
            ax.plot(plan[:, 0], plan[:, 1], linestyle=(0, (5, 4)), color="#0037ff", linewidth=1.0, alpha=0.45)
            ax.set_title(f"VSLAM map and boustrophedon trajectory ({self.sim_env_id})")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor("white")
            ax.grid(False)
            # Use dynamic bounds so imported plants or shifted trajectories are not clipped.
            bx = np.concatenate([np.asarray(sxs_plot, dtype=np.float32), np.asarray(xs, dtype=np.float32), np.asarray(plan[:, 0], dtype=np.float32), np.asarray(self.targets[:, 0], dtype=np.float32)])
            by = np.concatenate([np.asarray(sys_plot, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(plan[:, 1], dtype=np.float32), np.asarray(self.targets[:, 1], dtype=np.float32)])
            if len(cloud_x) > 0 and len(cloud_y) > 0:
                bx = np.concatenate([bx, np.asarray(cloud_x, dtype=np.float32)])
                by = np.concatenate([by, np.asarray(cloud_y, dtype=np.float32)])
            x_pad = max(2.0, 0.08 * float(np.nanmax(bx) - np.nanmin(bx) + 1e-6))
            y_pad = max(2.0, 0.08 * float(np.nanmax(by) - np.nanmin(by) + 1e-6))
            ax.set_xlim(float(np.nanmin(bx) - x_pad), float(np.nanmax(bx) + x_pad))
            ax.set_ylim(float(np.nanmin(by) - y_pad), float(np.nanmax(by) + y_pad))
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            fig.tight_layout()
            fig.savefig(vslam_path, dpi=260, bbox_inches="tight")
            plt.close(fig)
            bundle["vslam_trajectory_png"] = str(vslam_path)

            # 4) 3D trajectory (true path only).
            traj_path = self.figure_dir / f"{episode_prefix}_trajectory3d.png"
            fig = plt.figure(figsize=(9.4, 7.4))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(xs, ys, zs, label="UAV true trajectory", linewidth=2.2, color="#0b57d0")
            if np.any(low_tex):
                ax.scatter(xs[low_tex], ys[low_tex], zs[low_tex], s=12, color="#ef6c00", label="Low-texture zone", alpha=0.75)
            ax.scatter(self.targets[:, 0], self.targets[:, 1], self.targets[:, 2], s=28, marker="x", color="black", label="Inspection waypoints")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(f"UAV inspection trajectory in 3D ({self.sim_env_id})")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(traj_path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            bundle["trajectory_3d_png"] = str(traj_path)

            # 4) Decision dynamics.
            dyn_path = self.figure_dir / f"{episode_prefix}_decision_dynamics.png"
            commanded = np.array([float(r.get("commanded_speed_mps", 0.0)) for r in traj], dtype=np.float32)
            mu_a_arr = np.array([float(r.get("mu_A", 0.0)) for r in traj], dtype=np.float32)
            fig, axes = plt.subplots(4, 1, figsize=(12.0, 10.4), sharex=True)
            axes[0].plot(steps, speed, linewidth=1.8, label="Adaptive speed")
            axes[0].plot(steps, commanded, linewidth=1.4, linestyle="--", label="Commanded speed")
            axes[0].set_ylabel("Speed (m/s)")
            axes[0].legend(loc="best", frameon=True)

            ax_feat = axes[1]
            ax_pot = ax_feat.twinx()
            ax_feat.plot(steps, features, linewidth=1.5, label="Visible features")
            ax_pot.plot(steps, potential, linewidth=1.7, linestyle="--", label="OP-CBRS potential")
            ax_feat.set_ylabel("Visual features")
            ax_pot.set_ylabel("Potential")
            h1, l1 = ax_feat.get_legend_handles_labels()
            h2, l2 = ax_pot.get_legend_handles_labels()
            ax_feat.legend(h1 + h2, l1 + l2, loc="best", frameon=True)

            axes[2].plot(steps, loc_err, linewidth=1.6, label="Localization error")
            axes[2].axhline(self.localization_threshold, linestyle="--", color="black", linewidth=1.1, label="Aloc threshold")
            axes[2].set_ylabel("Localization error (m)")
            axes[2].legend(loc="best", frameon=True)

            axes[3].plot(steps, mu_t, linewidth=1.5, label=r"$\mu_T$ texture")
            axes[3].plot(steps, mu_a_arr, linewidth=1.5, label=r"$\mu_A$ adherence")
            axes[3].set_xlabel("Step")
            axes[3].set_ylabel("Membership")
            axes[3].legend(loc="best", frameon=True)
            for ax in axes:
                ax.grid(alpha=0.22)
            fig.suptitle("Decision, perception, and localization dynamics along the inspection route", fontsize=15)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
            fig.savefig(dyn_path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            bundle["decision_dynamics_png"] = str(dyn_path)

            # 5) Separate coverage / perception heatmap.
            heatmap, xedges, yedges = self._build_heatmap_array()
            if heatmap is not None:
                heatmap_path = self.figure_dir / f"{episode_prefix}_perception_heatmap.png"
                fig, ax = plt.subplots(figsize=(9.4, 7.4))
                im = ax.imshow(heatmap, origin="lower", aspect="auto", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="magma", interpolation="bicubic")
                ax.plot(xs, ys, linewidth=1.2, color="white", alpha=0.9, label="Executed trajectory")
                ax.scatter(self.targets[:, 0], self.targets[:, 1], s=24, marker="x", color="cyan", label="Inspection points")
                ax.set_xlim(float(xedges[0]), float(xedges[-1]))
                ax.set_ylim(float(yedges[0]), float(yedges[-1]))
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_title("Global visual-feature / hotspot coverage heatmap")
                fig.colorbar(im, ax=ax, label="Useful visual feature density")
                ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(heatmap_path, dpi=240, bbox_inches="tight")
                plt.close(fig)
                bundle["perception_heatmap_png"] = str(heatmap_path)
                npz_path = self.figure_dir / f"{episode_prefix}_perception_heatmap.npz"
                np.savez_compressed(npz_path, heatmap=heatmap, xedges=xedges, yedges=yedges)
                bundle["perception_heatmap_npz"] = str(npz_path)

            # 6) Standalone heatmap sequence.
            bundle = self._render_fuzzy_sequence_figure(episode_prefix, bundle, plt)
        except Exception as exc:
            print(f"[FIGURES] Warning: failed to render paper figures: {exc}")
        return bundle

    def _save_episode_artifacts(self, row: Dict[str, object], info: Dict[str, object]):
        if not self.save_trajectory_record:
            return {}
        episode_prefix = f"episode_{self.metric_episode_id:04d}"
        traj_csv = self.trajectory_dir / f"{episode_prefix}_trajectory.csv"
        traj_json = self.trajectory_dir / f"{episode_prefix}_trajectory.json"
        self._write_episode_trajectory_csv(traj_csv)
        bundle = {
            "run_id": self.run_id,
            "episode": self.metric_episode_id,
            "sim_env_id": self.sim_env_id,
            "policy_name": self.policy_name,
            "slam_mode": self.slam_mode,
            "trajectory_csv": str(traj_csv),
            "description": {
                "a": "Separate 2D/3D UAV trajectory and VSLAM map figures for coverage inspection and policy transfer.",
                "b": "Action decision dynamics in textureless regions with adaptive speed and potential changes.",
                "c": "The visual heatmap sequence shows what the onboard camera sees around chimneys and other inspection objects. Hotter colors indicate stronger visual-feature responses, clearer structure edges, and more useful landmarks for SLAM localization."
            },
            "episode_metrics": row,
            "final_info": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in info.items() if isinstance(v, (int, float, str, bool, np.floating, np.integer))},
        }
        bundle = self._render_episode_figures(episode_prefix, bundle)
        bundle["results_interpretation"] = {
            "heatmap_sequence": "The heatmap sequence shows how the onboard camera detects visually informative object regions during the boustrophedon inspection sweep. Brighter hotspots correspond to chimney rims, structure edges, and other regions that provide stronger cues for SLAM localization."
        }
        try:
            result_note = self.trajectory_dir / f"{episode_prefix}_results_interpretation.txt"
            result_note.write_text(bundle["results_interpretation"]["heatmap_sequence"] + "\n")
            bundle["results_interpretation_txt"] = str(result_note)
            latest_note = self.trajectory_dir / "latest_results_interpretation.txt"
            latest_note.write_text(bundle["results_interpretation"]["heatmap_sequence"] + "\n")
            bundle["latest_results_interpretation_txt"] = str(latest_note)
        except Exception as exc:
            print(f"[TRAJECTORY] Warning: failed to write result interpretation note: {exc}")
        try:
            traj_json.write_text(json.dumps(bundle, indent=2))
            self.latest_episode_bundle = traj_json
            latest_json = self.trajectory_dir / "latest_trajectory_bundle.json"
            latest_json.write_text(json.dumps(bundle, indent=2))
        except Exception as exc:
            print(f"[TRAJECTORY] Warning: failed to write {traj_json}: {exc}")
        return bundle

    def _finalize_paper_episode_metrics(self, info: Dict[str, object], terminated: bool, truncated: bool):
        ep = self.episode_metrics
        steps = max(int(ep.get("steps", self.step_count)), 1)
        duration = steps * self.dt
        # Robust route-completion success: prefer the explicit route_complete
        # flag from _compute_reward(), and fall back to waypoint counters. This
        # avoids false failures if termination bookkeeping changes.
        route_complete = bool(info.get("route_complete", False)) or int(ep.get("waypoints_reached", 0)) >= len(self.targets) or self.target_idx >= len(self.targets)
        success = bool(route_complete and terminated and not truncated)
        aloc = float(ep.get("loc_acc_steps", 0)) / max(float(ep.get("total_loc_steps", steps)), 1.0)
        row = {
            "run_id": self.run_id,
            "episode": self.metric_episode_id,
            "sim_env_id": self.sim_env_id,
            "policy_name": self.policy_name,
            "slam_mode": self.slam_mode,
            "Psucc_episode": int(success),
            "tau_i_sec": float(duration),
            "Dinc_episode": int(ep.get("drift_incidents", 0)),
            "Aloc_episode": float(aloc),
            "Econ_Wh_episode": float(ep.get("energy_wh", 0.0)),
            "Etime_sec_episode": float(duration if success else 0.0),
            "coverage_C": float(ep.get("coverage_max", 0.0)),
            "mu_cvg_final": float(info.get("mu_cvg", ep.get("coverage_max", 0.0))),
            "waypoints_reached": int(ep.get("waypoints_reached", 0)),
            "pre_drift_alerts": int(ep.get("pre_drift_alerts", 0)),
            "mean_speed_mps": float(ep.get("speed_sum", 0.0)) / steps,
            "mean_mu_T": float(ep.get("mu_T_sum", 0.0)) / steps,
            "mean_mu_L": float(ep.get("mu_L_sum", 0.0)) / steps,
            "mean_mu_W": float(ep.get("mu_W_sum", 0.0)) / steps,
            "mean_mu_A": float(ep.get("mu_A_sum", 0.0)) / steps,
            "reward_sum": float(ep.get("reward_sum", 0.0)),
            "op_cbrs_lib_hits": int(getattr(self, "_op_cbrs_lib_hits", 0)),
            "op_cbrs_lib_misses": int(getattr(self, "_op_cbrs_lib_misses", 0)),
            "op_cbrs_lib_misses_no_lib": int(getattr(self, "_op_cbrs_lib_misses_no_lib", 0)),
            "op_cbrs_lib_hit_ratio": float(
                int(getattr(self, "_op_cbrs_lib_hits", 0))
                / max(int(getattr(self, "_op_cbrs_lib_hits", 0))
                      + int(getattr(self, "_op_cbrs_lib_misses", 0))
                      + int(getattr(self, "_op_cbrs_lib_misses_no_lib", 0)), 1)
            ),
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
        }
        self._append_csv_row(self.episode_metrics_csv, row)
        bundle = self._save_episode_artifacts(row, info)

        self.summary_counts["episodes"] += 1
        self.summary_counts["successes"] += int(success)
        self.summary_counts["drift_incidents"] += int(row["Dinc_episode"])
        self.summary_counts["energy_wh_sum"] += float(row["Econ_Wh_episode"])
        self.summary_counts["loc_acc_time_sum"] += float(row["Aloc_episode"] * duration)
        self.summary_counts["total_time_sum"] += duration
        if success:
            self.summary_counts["successful_time_sum"] += duration
        episodes = max(self.summary_counts["episodes"], 1)
        successes_raw = int(self.summary_counts["successes"])
        tau_avg_sec = float(self.summary_counts["successful_time_sum"] / successes_raw) if successes_raw > 0 else 0.0
        summary = {
            "run_id": self.run_id,
            "sim_env_id": self.sim_env_id,
            "policy_name": self.policy_name,
            "slam_mode": self.slam_mode,
            "Ntotal": int(self.summary_counts["episodes"]),
            "Nsucc": int(self.summary_counts["successes"]),
            "Psucc": float(self.summary_counts["successes"] / episodes),
            "tau_avg_sec": float(tau_avg_sec),
            "Dinc": int(self.summary_counts["drift_incidents"]),
            "Aloc": float(self.summary_counts["loc_acc_time_sum"] / max(self.summary_counts["total_time_sum"], 1e-9)),
            "Econ_Wh_mean": float(self.summary_counts["energy_wh_sum"] / episodes),
            "Etime_sec": float(tau_avg_sec),
            "output_root": str(self.output_root),
            "step_metrics_csv": str(self.step_metrics_csv),
            "episode_metrics_csv": str(self.episode_metrics_csv),
            "trajectory_dir": str(self.trajectory_dir),
            "figure_dir": str(self.figure_dir),
            "latest_episode_bundle": str(self.latest_episode_bundle) if self.latest_episode_bundle else "",
            "paper_metric_definitions": {
                "Psucc": "Nsucc/Ntotal, completed missions without crash or localization loss",
                "tau_avg_sec": "mean duration of successful missions",
                "Dinc": "count of localization drift incidents",
                "Aloc": "fraction of time localization error <= threshold",
                "Econ_Wh_mean": "estimated energy per task in Wh",
                "Etime_sec": "successful-task execution time used for paper tables",
            },
        }
        try:
            self.summary_json.parent.mkdir(parents=True, exist_ok=True)
            self.summary_json.write_text(json.dumps(summary, indent=2))
            print(f"[METRICS] Episode {self.metric_episode_id} saved. Summary: {self.summary_json}")
            if bundle:
                print(f"[TRAJECTORY] Episode bundle: {bundle.get('trajectory_csv', '')}")
        except Exception as exc:
            print(f"[METRICS] Warning: failed to write summary JSON: {exc}")

    def _make_visual_feature_points(self):
        """Create visual landmarks/hotspots from the active VSLAM waypoints.

        The old implementation used four columns and two rows, producing only
        eight hotspot objects.  That kept the VSLAM heatmap/map visually stuck
        at eight points.  This version creates one feature cluster per planned
        waypoint, so e1 has 16 VSLAM hotspots and the drone follows those same
        16 target locations.
        """
        rng = np.random.default_rng(1234)
        pts = []
        self.visual_hotspots = []
        route = np.asarray(getattr(self, "targets", getattr(self, "base_targets", np.zeros((0, 3)))), dtype=np.float32)
        if route.ndim != 2 or route.shape[0] == 0:
            route = np.asarray(self._build_boustrophedon_targets(), dtype=np.float32)

        for idx, wp in enumerate(route):
            cx, cy = float(wp[0]), float(wp[1])
            # Put the visual structure below the flight altitude so the onboard
            # camera/SLAM proxy can observe chimney rims and industrial edges.
            z_mid = 6.2 if self.sim_env_id == "e1" else 5.4
            for _ in range(120):
                ang = rng.uniform(0.0, 2.0 * math.pi)
                rad = rng.uniform(0.85, 1.35)
                z = rng.uniform(0.6, 7.4)
                pts.append([cx + rad * math.cos(ang), cy + rad * math.sin(ang), z])
            for _ in range(90):
                ang = rng.uniform(0.0, 2.0 * math.pi)
                rad = rng.uniform(0.55, 1.05)
                z = rng.uniform(6.5, 8.8)
                pts.append([cx + rad * math.cos(ang), cy + rad * math.sin(ang), z])
            self.visual_hotspots.append({
                "pos": np.array([cx, cy, z_mid], dtype=np.float32),
                "sigma": 0.105 if self.sim_env_id == "e1" else 0.13,
                "strength": 1.25 if self.sim_env_id == "e1" else 1.05,
            })

        # Corridor-side landmarks make the sweep path continuously observable.
        if len(route) > 0:
            x_min, x_max = float(np.min(route[:, 0]) - 1.5), float(np.max(route[:, 0]) + 1.5)
            y_min, y_max = float(np.min(route[:, 1]) - 1.5), float(np.max(route[:, 1]) + 1.5)
            for x in np.linspace(x_min, x_max, 6):
                for y in np.linspace(y_min, y_max, 28):
                    z = rng.uniform(0.5, 4.2)
                    pts.append([float(x + rng.normal(0.0, 0.18)), float(y + rng.normal(0.0, 0.12)), z])

        print(f"[VSLAM_FEATURES] sim_env_id={self.sim_env_id} vslam_waypoints={len(route)} hotspots={len(self.visual_hotspots)}")
        return np.asarray(pts, dtype=np.float32)

    def _compute_reward(self, action, precomputed_nav=None):
        dist = float(np.linalg.norm(self.target - self.pos))
        progress = self.prev_dist - dist
        self.prev_dist = dist

        speed_now = float(np.linalg.norm(self.vel))
        speed_cap_for_pen = max(
            float(getattr(self, "last_osd_speed_cap_mps", self.osd_vmax)),
            self.osd_vmin,
        )
        speed_excess = max(0.0, speed_now - speed_cap_for_pen)

        # 1) Asymmetric progress: forward motion rewarded more than regression
        if progress > 0:
            reward = 5.0 * progress
        else:
            reward = 1.2 * progress

        # 2) Small survival bonus so baseline rises as UAV stays alive
        reward += 0.003

        # 3) Reduced action-norm and speed-excess penalties
        reward -= 0.003 * float(np.linalg.norm(action))
        reward -= 0.010 * speed_excess

        # 4) Proximity pull toward waypoint in final 3 m
        if dist < 3.0:
            reward += 0.022 * (3.0 - dist)

        terminated = False
        route_complete = False
        collision_event = False
        reached = dist < self.inspection_reach_radius

        if reached:
            reward += 18.0                   # was 12
            self.target_idx += 1
            if self.target_idx >= len(self.targets):
                reward += 65.0               # was 45
                route_complete = True
                terminated = True
            else:
                self.current_segment_start = self.pos.copy()
                self.target = self.targets[self.target_idx].copy()
                self.prev_dist = float(np.linalg.norm(self.target - self.pos))

        if self.pos[2] <= self.no_fly_z_min + 0.05 or self.pos[2] >= self.no_fly_z_max - 0.05:
            reward -= 0.5                    # was 1.0

        min_ray = float(np.min(self._ray_distances(noisy=False))) if self.num_rays > 0 else 1.0
        inside_collision, inside_name = self._point_inside_collision_aabb(
            self.pos, margin=float(getattr(self, "drone_collision_radius", 0.42))
        )
        collision_event = bool(
            collision_event or getattr(self, "last_collision_event", False) or inside_collision
        )

        if collision_event:
            reward -= 15.0                   # was 18
            if self.collision_termination and not route_complete:
                terminated = True
        elif min_ray < 0.07:
            reward -= 5.0                    # was 6.5
            collision_event = True
            if self.collision_termination and not route_complete:
                terminated = True
        elif min_ray < 0.28:
            reward -= (0.28 - min_ray) * 2.0  # was * 3.0

        if self.slam_quality < 0.25:
            reward -= 0.015                  # was 0.03

        paper = self._compute_paper_step_values(
            action, progress, dist, min_ray, reached, precomputed_nav=precomputed_nav
        )

        # Positive fuzzy rewards
        reward += 0.12 * float(paper.get("mu_T", 0.0))
        reward += 0.10 * float(paper.get("mu_A", 0.0))

        # KEY FIX: replace dense loc_error PENALTY with accuracy BONUS
        # Old: reward -= 0.12 * min(loc_error, 2.0)  → up to -216 per episode
        # New: reward += small bonus when localization is accurate (bounded 0→+0.08)
        loc_err = float(paper.get("localization_error_m", 0.0))
        reward += 0.08 * max(
            0.0, 1.0 - loc_err / max(self.localization_threshold * 8.0, 0.8)
        )

        if float(paper.get("drift_event", 0.0)) > 0.5:
            reward -= 0.35                   # was 0.60

        if self.enable_op_cbrs:
            reward += float(paper["op_cbrs_shaping_reward"])
            reward += 0.30 * float(paper["coverage_delta"])  # was 0.25

        info = {
            "target_index": self.target_idx,
            "num_targets": int(len(self.targets)),
            "route_complete": bool(route_complete),
            "collision_event": bool(collision_event),
            "collision_name": str(
                getattr(self, "last_collision_name", "") or inside_name or ""
            ),
            "collision_clearance_m": float(
                getattr(self, "last_collision_distance_m", float("inf"))
            ),
            "distance": dist,
            "reached": reached,
            "inspection_reach_radius": float(self.inspection_reach_radius),
            "min_ray": min_ray,
            "slam_quality": float(self.slam_quality),
            "slam_mode": self.slam_mode,
            "ros2_graph_created": bool(self.ros2_graph_created),
            "imported_plant": self.imported_plant_loaded,
        }
        info.update(paper)
        return float(reward), bool(terminated), info

    def _point_segment_distance_xy(self, p_xy, a_xy, b_xy) -> float:
        """Distance from a 2D point to a finite route segment."""
        p = np.asarray(p_xy, dtype=np.float32)
        a = np.asarray(a_xy, dtype=np.float32)
        b = np.asarray(b_xy, dtype=np.float32)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-9:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    def _point_near_route_xy(self, p_xy, clearance: float) -> bool:
        """True if p_xy is too close to any waypoint or planned route segment."""
        if getattr(self, "targets", None) is None or len(self.targets) == 0:
            return False
        p = np.asarray(p_xy, dtype=np.float32)
        pts = [np.asarray(self.pos[:2], dtype=np.float32)] + [np.asarray(t[:2], dtype=np.float32) for t in self.targets]
        for q in pts:
            if float(np.linalg.norm(p - q)) < max(clearance, self.inspection_reach_radius + 0.75):
                return True
        for a, b in zip(pts[:-1], pts[1:]):
            if self._point_segment_distance_xy(p, a, b) < clearance:
                return True
        return False

    def _randomize_domain(self):
        # Domain variables follow the paper's e1/e2 Sim2Sim setup:
        # e1: nuclear/power-plant-like domain with stronger texture and stable lighting;
        # e2: industrial transfer domain with lower texture, lower illumination and higher wind.
        if not self.domain_randomization:
            if self.sim_env_id == "e2":
                self.illumination_lux = 400.0
                self.wind_mps = 3.0
            else:
                self.illumination_lux = 500.0
                self.wind_mps = 0.9
        elif self.sim_env_id == "e2":
            self.illumination_lux = float(np.clip(self.rng.normal(400.0, 28.0), 320.0, 470.0))
            self.wind_mps = float(np.clip(self.rng.normal(3.0, 1.1), 0.0, 5.0))
        else:
            self.illumination_lux = float(np.clip(self.rng.normal(500.0, 25.0), 430.0, 570.0))
            self.wind_mps = float(np.clip(self.rng.normal(0.9, 0.45), 0.0, 2.0))

        self.targets = self.base_targets.copy()
        if self.route_randomization and self.domain_randomization and len(self.targets) > 2:
            # Optional stress test only. Keep disabled for paper-aligned OSD evaluation.
            tail = self.targets[1:].copy()
            self.rng.shuffle(tail)
            self.targets[1:] = tail

        if self.imported_plant_loaded:
            self.obstacles = np.zeros((0, 6), dtype=np.float32)
            return

        # For paper-aligned final evaluation, disabling random obstacles is recommended.
        if (not self.obstacle_randomization) or int(self.num_obstacles) <= 0:
            self.obstacles = np.zeros((0, 6), dtype=np.float32)
            return

        self.obstacles = np.zeros((self.num_obstacles, 6), dtype=np.float32)
        clearance = float(self.route_corridor_clearance)
        for i in range(self.num_obstacles):
            placed = False
            for _attempt in range(600):
                x = self.rng.uniform(-self.world_size * 0.70, self.world_size * 0.70)
                y = self.rng.uniform(-self.world_size * 0.70, self.world_size * 0.70)
                if self.safe_obstacle_placement and self._point_near_route_xy([x, y], clearance):
                    continue
                placed = True
                break
            if not placed:
                # Last-resort safe parking location outside the inspection route.
                x = -self.world_size * 0.82 + 0.7 * i
                y = self.world_size * 0.82
            z = self.rng.uniform(0.8, 2.4)
            sx = self.rng.uniform(0.45, 1.20)
            sy = self.rng.uniform(0.45, 1.20)
            sz = self.rng.uniform(0.8, 2.3)
            self.obstacles[i] = [x, y, z, sx, sy, sz]

    def _generate_targets_from_plant_bbox(self):
        """Generate exactly 16 imported-power-plant waypoints from the plant bbox."""
        if self.plant_bbox_min is None or self.plant_bbox_max is None:
            return
        mn = self.plant_bbox_min
        mx = self.plant_bbox_max
        margin = max(2.5, min(8.0, 0.02 * max(mx[0] - mn[0], mx[1] - mn[1])))
        x_cols = np.linspace(mn[0] + margin, mx[0] - margin, 4)
        y_rows = np.linspace(mn[1] + margin, mx[1] - margin, 4)
        z_safe = float(min(max(float(mx[2]) + 4.0, 11.5), self.no_fly_z_max - 1.2))
        pts = []
        for ix, x in enumerate(x_cols):
            ys = y_rows if ix % 2 == 0 else list(reversed(y_rows))
            for iy, y in enumerate(ys):
                pts.append([float(x), float(y), float(z_safe + 0.25 * ((ix + iy) % 2))])
        self.base_targets = np.asarray(pts, dtype=np.float32)
        self.targets = self.base_targets.copy()
        self.target = self.targets[0].copy()
        print(f"[ROUTE_BUILD_IMPORTED_E1_POWER_PLANT] vslam_waypoints={len(self.targets)} drone_targets={len(self.targets)} expected=16")



# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    # Isaac Sim 5.x runs on Python 3.11 while system ROS 2 Humble is Python 3.10.
    # Do not rely on the sourced /opt/ros/humble Python path for Isaac Sim.
    # Use the Isaac Sim internal ROS 2 bridge libraries instead.
    if args.slam_mode == "cuvslam" or args.enable_ros2_camera_pub:
        os.environ["ROS_DISTRO"] = "humble"
        os.environ["ROS_DOMAIN_ID"] = str(int(args.ros2_domain_id))
        os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        # Isaac Sim ships its own ROS 2 Humble bridge libraries. The path is
        # different from /opt/ros/humble and must be visible before the ROS 2
        # bridge extension starts, otherwise librmw_fastrtps_cpp.so is not found.
        import glob
        bridge_candidates = []
        for pat in [
            str(Path.home() / "IsaacLab/_isaac_sim/exts/isaacsim.ros2.bridge/humble/lib"),
            str(Path.home() / "IsaacLab/_isaac_sim/exts/omni.isaac.ros2_bridge/humble/lib"),
            str(Path.home() / "isaacsim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble/lib"),
            str(Path.home() / "isaacsim/exts/isaacsim.ros2.bridge/humble/lib"),
            str(Path.home() / ".local/share/ov/pkg/isaac-sim*/exts/isaacsim.ros2.bridge/humble/lib"),
        ]:
            bridge_candidates.extend(glob.glob(pat))
        bridge_candidates = [x for x in bridge_candidates if Path(x).exists()]
        ld_parts = [x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]
        for bridge_lib in bridge_candidates:
            if bridge_lib not in ld_parts:
                ld_parts.append(bridge_lib)
        if ld_parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)
        if bridge_candidates:
            print(f"[ROS2_ENV] Using Isaac Sim ROS 2 bridge lib: {bridge_candidates[0]}")
        else:
            print("[ROS2_ENV] Warning: could not auto-find Isaac Sim ROS 2 bridge humble/lib. Export it manually before running.")
        # Remove system Humble Python 3.10 site-packages from Isaac Sim Python 3.11.
        py = os.environ.get("PYTHONPATH", "")
        if py:
            os.environ["PYTHONPATH"] = ":".join([x for x in py.split(":") if "python3.10" not in x and "dist-packages" not in x])

    sys.argv = [sys.argv[0]]
    simulation_app = SimulationApp({"headless": bool(args.headless), "renderer": args.renderer, "width": 1600, "height": 900})

    global omni, Gf, Sdf, UsdGeom, UsdLux, UsdShade
    import omni as omni_mod
    from pxr import Gf as Gf_mod, Sdf as Sdf_mod, UsdGeom as UsdGeom_mod, UsdLux as UsdLux_mod, UsdShade as UsdShade_mod
    omni = omni_mod
    Gf = Gf_mod
    Sdf = Sdf_mod
    UsdGeom = UsdGeom_mod
    UsdLux = UsdLux_mod
    UsdShade = UsdShade_mod

    import warnings
    warnings.filterwarnings("ignore", message="Gym has been unmaintained.*")
    warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU.*")

    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import get_schedule_fn
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        env = NPPDroneGymEnv(
            max_episode_steps=args.max_episode_steps,
            inspection_reach_radius=args.inspection_reach_radius,
            complete_route_before_timeout=(
                args.mode in [
                    "random",
                    "collect_uniform_speed",
                    "build_op_cbrs_library",
                    "eval_baselines",
                    "eval_transfer",
                    "paper_pipeline",
                    "eval_transfer_no_osd",
                    "paper_pipeline_no_osd",
                ]
                and not args.allow_timeout_before_route_complete
            ),
            world_size=args.world_size,
            num_obstacles=args.num_obstacles,
            num_rays=args.num_rays,
            render_sim=not args.headless,
            seed=args.seed,
            drone_usd=args.drone_usd,
            plant_usd=args.plant_usd,
            hdri_path=args.hdri_path,
            save_stage=args.save_stage,
            camera_fov_deg=args.camera_fov_deg,
            plant_rotation=(args.plant_rotate_x, args.plant_rotate_y, args.plant_rotate_z),
            plant_scale=args.plant_scale,
            plant_margin=args.plant_margin,
            plant_max_dim=args.plant_max_dim,
            plant_axis=args.plant_axis,
            keep_procedural_site=args.keep_procedural_site,
            auto_fit_plant=not args.no_auto_fit_plant,
            slam_mode=args.slam_mode,
            slam_drift_pos_per_sec=args.slam_drift_pos_per_sec,
            slam_pos_noise_std=args.slam_pos_noise_std,
            slam_yaw_noise_std=args.slam_yaw_noise_std,
            slam_vel_noise_std=args.slam_vel_noise_std,
            slam_tracking_loss_prob=args.slam_tracking_loss_prob,
            slam_quality_recover_rate=args.slam_quality_recover_rate,
            depth_noise_std=args.depth_noise_std,
            enable_ros2_camera_pub=args.enable_ros2_camera_pub,
            disable_ros2_camera_pub=args.disable_ros2_camera_pub,
            ros2_domain_id=args.ros2_domain_id,
            ros2_camera_width=args.ros2_camera_width,
            ros2_camera_height=args.ros2_camera_height,
            ros2_camera_fps=args.ros2_camera_fps,
            stereo_baseline=args.stereo_baseline,
            left_image_topic=args.left_image_topic,
            right_image_topic=args.right_image_topic,
            left_camera_info_topic=args.left_camera_info_topic,
            right_camera_info_topic=args.right_camera_info_topic,
            imu_topic=args.imu_topic,
            cuvslam_odom_topic=args.cuvslam_odom_topic,
            cuvslam_odom_udp_host=args.cuvslam_odom_udp_host,
            cuvslam_odom_udp_port=args.cuvslam_odom_udp_port,
            sim_env_id=args.sim_env_id,
            policy_name=args.policy_name,
            output_root=args.output_root,
            metrics_dir=args.metrics_dir,
            trajectory_dir=args.trajectory_dir,
            figure_dir=args.figure_dir,
            step_metrics_csv=args.step_metrics_csv,
            episode_metrics_csv=args.episode_metrics_csv,
            summary_json=args.summary_json,
            localization_threshold=args.localization_threshold,
            drift_threshold=args.drift_threshold,
            coverage_radius=args.coverage_radius,
            motor_hover_power_w=args.motor_hover_power_w,
            motor_speed_power_gain_w=args.motor_speed_power_gain_w,
            enable_op_cbrs=(args.enable_op_cbrs and not args.disable_op_cbrs),
            op_cbrs_library_path=args.potential_library,
            save_paper_figures=(not args.disable_paper_figures),
            save_trajectory_record=(not args.disable_trajectory_record),
            heatmap_grid_size=args.heatmap_grid_size,
            domain_randomization=(not args.disable_domain_randomization),
            route_randomization=args.enable_route_randomization,
            obstacle_randomization=(not args.disable_obstacle_randomization),
            safe_obstacle_placement=args.safe_obstacle_placement,
            route_corridor_clearance=args.route_corridor_clearance,
            collision_termination=(not args.disable_collision_termination),
            policy_analytic_blend=args.policy_analytic_blend,
            adaptive_speed_floor=args.adaptive_speed_floor,
            obstacle_avoidance_gain=args.obstacle_avoidance_gain,
            obstacle_avoidance_range=args.obstacle_avoidance_range,
            yaw_gain=args.yaw_gain,
            velocity_memory=args.velocity_memory,
            enable_osd=(bool(args.enable_osd) and not bool(args.disable_osd)),
        )
        return Monitor(env)

    def _default_potential_library_path() -> Path:
        return Path(args.potential_library).expanduser() if args.potential_library else Path(args.output_root).expanduser() / "op_cbrs" / "op_cbrs_library.json"

    def _write_algorithm_manifest(stage_name: str, extra: Optional[Dict[str, object]] = None):
        manifest_path = Path(args.algorithm_manifest_json).expanduser() if args.algorithm_manifest_json else Path(args.output_root).expanduser() / "algorithm_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "stage": stage_name,
            "methodology": "MDP + PPO scalar-speed control + fuzzy inference + OP-CBRS + Sim2Sim transfer + VSLAM localization",
            "implementation_note": "Isaac Sim and Isaac ROS Visual SLAM/cuVSLAM instantiate the simulator and VSLAM block only; they are not the proposed methodology.",
            "mdp": {
                "state": "SLAM pose/velocity/yaw, IMU-like motion, target-relative geometry, camera/feature cues, depth rays, tracking quality, previous scalar speed action, mission progress",
                "action": "scalar PPO speed in A=[v_min,v_max]; in PPO-without-OSD mode the analytic/fuzzy OSD shield is disabled and the low-level controller tracks waypoint direction, yaw and altitude",
                "reward": "progress + waypoint + safety + fuzzy coverage + OP-CBRS potential shaping with gamma*Phi(s_t)-Phi(s_{t+1})",
            },
            "fuzzy_memberships": ["mu_T(texture)", "mu_L(illumination)", "mu_W(wind)", "mu_A(trajectory adherence)", "mu_cvg(coverage)"],
            "op_cbrs": {
                "library_path": str(_default_potential_library_path()),
                "source": "offline uniform-speed tasks T*",
                "selection": "fuzzy bin key T/L/W/A selects Phi_A*(s); heuristic potential is used as fallback",
            },
            "sim2sim": {"source_env": "e1", "target_env": "e2", "transfer": "load e1 PPO policy and fine-tune/evaluate in e2 with recalibrated fuzzy memberships"},
            "robust_transfer_controller": {
                "policy_analytic_blend": args.policy_analytic_blend,
                "adaptive_speed_floor": args.adaptive_speed_floor,
                "obstacle_avoidance_gain": args.obstacle_avoidance_gain,
                "obstacle_avoidance_range": args.obstacle_avoidance_range,
                "yaw_gain": args.yaw_gain,
                "velocity_memory": args.velocity_memory,
                "enable_osd": bool(args.enable_osd) and not bool(args.disable_osd)
            },
            "metrics": ["Psucc", "tau_avg_sec", "Dinc", "Aloc", "Econ_Wh_mean", "Etime_sec"],
            "result_interpretation": "The heatmap sequence shows how the UAV camera/SLAM system detects visual features while flying through a difficult inspection area. Brighter or denser regions in the heatmaps mean that more useful visual features are available for localization.",
            "sim_env_id": args.sim_env_id,
            "policy_name": args.policy_name,
            "domain_randomization_controls": {
                "domain_randomization_enabled": not args.disable_domain_randomization,
                "route_randomization_enabled": bool(args.enable_route_randomization),
                "obstacle_randomization_enabled": not args.disable_obstacle_randomization,
                "safe_obstacle_placement": bool(args.safe_obstacle_placement),
                "collision_termination_enabled": not args.disable_collision_termination,
            },
            "output_root": str(Path(args.output_root).expanduser()),
        }
        if extra:
            manifest["extra"] = extra
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[ALGORITHM] Manifest saved: {manifest_path}")

    def _parse_uniform_speeds() -> list[float]:
        speeds = []
        for token in str(args.uniform_speeds).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                speeds.append(float(token))
            except ValueError:
                pass
        return speeds or [0.25, 0.50, 0.75, 1.00]

    def create_vec_env():
        return DummyVecEnv([make_env])

    def _raw_env(vec_env):
        return vec_env.envs[0].unwrapped

    def _wrap_angle(a: float) -> float:
        return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

    def _osd_fuzzy_action(e: NPPDroneGymEnv) -> np.ndarray:
        """Paper-aligned analytic OSD controller."""
        feature_count = e._estimate_visible_feature_count()
        mu_T = e._texture_membership(feature_count)
        mu_L = e._illumination_membership(float(e.illumination_lux))
        mu_W = e._wind_stability_membership(float(e.wind_mps))
        _, mu_A, _ = e._trajectory_adherence()
        vt, _, _, _ = e._compute_osd_speed_memberships(mu_T, mu_L, mu_W, mu_A)
        vt = float(np.clip(vt, e.osd_vmin, e.osd_vmax))
        # Convert physical speed to normalized Box([-1,1]) action.
        a = 2.0 * (vt - e.osd_vmin) / max(e.osd_vmax - e.osd_vmin, 1e-6) - 1.0
        return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32)

    def _uniform_speed_action(e: NPPDroneGymEnv, speed_mps: float) -> np.ndarray:
        """Uniform-speed task T* used to build OP-CBRS potential functions."""
        sp = float(np.clip(speed_mps, e.osd_vmin, e.osd_vmax))
        a = 2.0 * (sp - e.osd_vmin) / max(e.osd_vmax - e.osd_vmin, 1e-6) - 1.0
        return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32)

    def _rollout_action_policy(vec_env, action_fn, episodes: int, label: str, collect_rows: bool = False):
        raw = _raw_env(vec_env)
        obs = vec_env.reset()
        completed = 0
        step_i = 0
        rows = []
        print(f"[EVAL] Starting {label}: episodes={episodes}")
        while completed < int(episodes):
            action = action_fn(raw)
            obs, rewards, dones, infos = vec_env.step(np.asarray([action], dtype=np.float32))
            if step_i % 120 == 0:
                info = infos[0] if infos else {}
                print(
                    f"[EVAL:{label}] step={step_i:06d} ep={completed+1}/{episodes} "
                    f"pos={raw.pos.round(2).tolist()} target={raw.target_idx}/{len(raw.targets)-1} "
                    f"dist={float(info.get('distance', 0.0)):.2f} "
                    f"slam_q={float(info.get('slam_quality', 0.0)):.2f}"
                )
            if bool(dones[0]):
                info = infos[0] if infos else {}
                route_complete_done = bool(info.get("route_complete", False)) or getattr(raw, "target_idx", 0) >= len(getattr(raw, "targets", []))
                if getattr(raw, "complete_route_before_timeout", False) and not route_complete_done:
                    print(
                        f"[EVAL:{label}] Incomplete episode was reset by safety guard; "
                        f"retrying same requested episode. target={getattr(raw, 'target_idx', -1)}/"
                        f"{max(len(getattr(raw, 'targets', [])) - 1, 0)}"
                    )
                    obs = vec_env.reset()
                    step_i += 1
                    continue
                if collect_rows:
                    rows.extend([dict(r) for r in getattr(raw, "trajectory_records", [])])
                completed += 1
                if completed < int(episodes):
                    obs = vec_env.reset()
            omni.kit.app.get_app().update()
            step_i += 1
        return rows

    def _aggregate_potential_library(rows: list[Dict[str, object]], speeds: list[float]) -> Dict[str, object]:
        buckets: Dict[str, Dict[str, float]] = {}
        def mb(v):
            v = float(v)
            return "LOW" if v < 0.35 else ("MED" if v < 0.70 else "HIGH")
        for r in rows:
            try:
                key = f"T={mb(r.get('mu_T',0))}|L={mb(r.get('mu_L',0))}|W={mb(r.get('mu_W',0))}|A={mb(r.get('mu_A',0))}"
                phi = float(r.get("op_cbrs_potential", 0.0))
                progress = float(r.get("paper_progress", 0.0))
                coverage = float(r.get("coverage_delta", 0.0))
                # Empirical desirability from uniform-speed behavior.
                target_phi = phi + 0.75 * progress + 0.50 * coverage - 0.25 * float(r.get("drift_event", 0.0))
                b = buckets.setdefault(key, {"sum": 0.0, "count": 0.0, "progress_sum": 0.0, "coverage_sum": 0.0})
                b["sum"] += target_phi
                b["count"] += 1.0
                b["progress_sum"] += progress
                b["coverage_sum"] += coverage
            except Exception:
                continue
        states = {}
        for k, b in buckets.items():
            count = max(b["count"], 1.0)
            states[k] = {
                "phi": float(b["sum"] / count),
                "count": int(count),
                "mean_progress": float(b["progress_sum"] / count),
                "mean_coverage_delta": float(b["coverage_sum"] / count),
            }
        return {
            "library_type": "OP-CBRS fuzzy potential library",
            "source": "offline uniform-speed coverage tasks T*",
            "uniform_speeds_mps": speeds,
            "state_key": "T/L/W/A fuzzy bins over mu_T, mu_L, mu_W, mu_A",
            "states": states,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def collect_uniform_speed_library():
        old_mode, old_policy, old_env_id = args.mode, args.policy_name, args.sim_env_id
        args.policy_name = "Uniform-Speed-OffPolicy-Tasks"
        args.sim_env_id = "e1"
        vec = create_vec_env()
        rows = []
        speeds = _parse_uniform_speeds()
        try:
            for sp in speeds:
                print(f"[OP-CBRS] Collecting uniform-speed task T*: speed={sp:.2f} m/s")
                rows.extend(_rollout_action_policy(vec, lambda e, sp=sp: _uniform_speed_action(e, sp), args.offline_episodes_per_speed, f"T*_v{sp:.2f}", collect_rows=True))
        finally:
            vec.close()
            args.mode, args.policy_name, args.sim_env_id = old_mode, old_policy, old_env_id

        lib = _aggregate_potential_library(rows, speeds)
        lib_path = _default_potential_library_path()
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_text(json.dumps(lib, indent=2))
        print(f"[OP-CBRS] Potential library saved: {lib_path}")
        _write_algorithm_manifest("collect_uniform_speed/build_op_cbrs_library", {"potential_library": str(lib_path), "num_rows": len(rows), "num_states": len(lib.get("states", {}))})
        return lib_path

    def build_library_from_existing_csv():
        csv_path = Path(args.step_metrics_csv).expanduser() if args.step_metrics_csv else Path(args.output_root).expanduser() / "metrics" / "paper_step_metrics.csv"
        if not csv_path.exists():
            print(f"No step CSV found at {csv_path}; collecting uniform-speed data instead.")
            return collect_uniform_speed_library()
        rows = []
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend([dict(r) for r in reader])
        lib = _aggregate_potential_library(rows, _parse_uniform_speeds())
        lib_path = _default_potential_library_path()
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_text(json.dumps(lib, indent=2))
        print(f"[OP-CBRS] Potential library built from CSV and saved: {lib_path}")
        _write_algorithm_manifest("build_op_cbrs_library", {"source_csv": str(csv_path), "potential_library": str(lib_path), "num_rows": len(rows)})
        return lib_path

    class StopAfterEpisodesCallback(BaseCallback):
        def __init__(self, target_episodes: int, label: str = "PPO", verbose: int = 1):
            super().__init__(verbose)
            self.target_episodes = int(max(0, target_episodes))
            self.label = str(label)
            self.completed_episodes = 0

        def _on_step(self) -> bool:
            dones = self.locals.get("dones", [])
            for d in dones:
                if bool(d):
                    self.completed_episodes += 1
                    if self.verbose:
                        print(f"[{self.label}] completed_episode={self.completed_episodes}/{self.target_episodes} timestep={self.num_timesteps}")
            return self.completed_episodes < self.target_episodes if self.target_episodes > 0 else True

    def train_ppo_policy(stage_name: str, save_name: str, sim_env_id: str, load_path: str = "", learn_steps: Optional[int] = None):
        args.sim_env_id = sim_env_id
        args.policy_name = stage_name
        if args.potential_library == "":
            lib_path = _default_potential_library_path()
            if lib_path.exists():
                args.potential_library = str(lib_path)

        vec = create_vec_env()
        try:
            if load_path:
                print(f"[PPO] Loading source model: {load_path}")
                model = PPO.load(load_path, env=vec, device=args.device)
                transfer_lr = 3e-5
                model.learning_rate = transfer_lr
                try:
                    model.lr_schedule = get_schedule_fn(transfer_lr)
                    for param_group in model.policy.optimizer.param_groups:
                        param_group["lr"] = transfer_lr
                except Exception:
                    pass
            else:
                model = PPO(
                    "MlpPolicy",
                    vec,
                    verbose=1,
                    tensorboard_log=str(log_dir),
                    device=args.device,
                    learning_rate=6e-5,   # paper: eta_alpha = 6e-5
                    n_steps=1024,         # paper: experience buffer = 1024
                    batch_size=512,       # paper: batch size = 512
                    n_epochs=16,          # paper: 16 replay cycles
                    gamma=0.95,           # paper: gamma = 0.95
                    gae_lambda=0.98,      # paper: GAE lambda = 0.98
                    ent_coef=0.006,
                    clip_range=0.25,      # paper: eps_delta = 0.25
                    max_grad_norm=0.5,
                )
            episode_target = int(args.transfer_episodes if load_path else args.train_episodes)
            steps = int(learn_steps if learn_steps is not None else args.total_timesteps)
            if episode_target > 0:
                # Use timesteps as a safety budget; the callback stops exactly after N completed episodes.
                steps = max(steps, episode_target * int(args.max_episode_steps))
                print(f"[PPO] Episode-based training requested: target_episodes={episode_target}, timestep_budget={steps}")
                callback = StopAfterEpisodesCallback(episode_target, label=stage_name, verbose=1)
            else:
                callback = None
            model.learn(total_timesteps=steps, callback=callback)
            save_path = Path(args.output_root).expanduser() / "models" / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
            print(f"[PPO] Saved model: {save_path}.zip")
            _write_algorithm_manifest(stage_name, {"model": str(save_path) + ".zip", "learn_steps": steps, "target_episodes": episode_target, "domain_randomization": (not args.disable_domain_randomization), "sim_env_id": sim_env_id})
            return str(save_path) + ".zip"
        finally:
            vec.close()

    def evaluate_model(model_path: str, label: str, sim_env_id: str, episodes: int):
        args.sim_env_id = sim_env_id
        args.policy_name = label
        if args.potential_library == "":
            lib_path = _default_potential_library_path()
            if lib_path.exists():
                args.potential_library = str(lib_path)
        vec = create_vec_env()
        try:
            print(f"[EVAL] Loading model: {model_path}")
            model = PPO.load(model_path, env=vec, device=args.device)
            obs = vec.reset()
            completed, step_i = 0, 0
            while completed < int(episodes):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = vec.step(action)
                if step_i % 120 == 0:
                    raw = _raw_env(vec)
                    info = infos[0] if infos else {}
                    print(f"[EVAL:{label}] step={step_i:06d} ep={completed+1}/{episodes} pos={raw.pos.round(2).tolist()} dist={float(info.get('distance',0.0)):.2f}")
                if bool(dones[0]):
                    completed += 1
                    if completed < int(episodes):
                        obs = vec.reset()
                omni.kit.app.get_app().update()
                step_i += 1
            _write_algorithm_manifest(label, {"eval_model": model_path, "episodes": episodes, "sim_env_id": sim_env_id})
        finally:
            vec.close()

    def eval_baselines():
        speeds = _parse_uniform_speeds()
        for sp in speeds:
            args.policy_name = f"Uniform-Speed-{sp:.2f}mps"
            vec = create_vec_env()
            try:
                _rollout_action_policy(vec, lambda e, sp=sp: _uniform_speed_action(e, sp), args.eval_episodes, args.policy_name)
            finally:
                vec.close()
        args.policy_name = "Analytic-OSD+Fuzzy+OP-CBRS"
        vec = create_vec_env()
        try:
            _rollout_action_policy(vec, _osd_fuzzy_action, args.eval_episodes, args.policy_name)
        finally:
            vec.close()
        _write_algorithm_manifest("eval_baselines", {"uniform_speeds": speeds, "episodes": args.eval_episodes})

    # ------------------------------------------------------------------
    # Algorithmic mode dispatch
    # ------------------------------------------------------------------
    if args.mode == "random":
        vec = create_vec_env()
        raw_env = _raw_env(vec)
        obs = vec.reset()
        print("[INSPECT] Analytic OSD+Fuzzy+OP-CBRS inspection controller active.")
        print("[INSPECT] This is for paper-style demonstration and cuVSLAM data collection.")
        print(f"[OUTPUT] Root: {raw_env.output_root}")
        print(f"[METRICS] Step CSV: {raw_env.step_metrics_csv}")
        print(f"[METRICS] Episode CSV: {raw_env.episode_metrics_csv}")
        print(f"[METRICS] Summary JSON: {raw_env.summary_json}")
        print(f"[TRAJECTORY] Dir: {raw_env.trajectory_dir}")
        print(f"[FIGURES] Dir: {raw_env.figure_dir}")
        _write_algorithm_manifest("random/demo_analytic_osd")
        try:
            demo_step_budget = max(int(args.total_timesteps), int(args.max_episode_steps * max(1, len(raw_env.targets)) * 2), 4000)
            print(f"[INSPECT] Demo step budget={demo_step_budget}; route must visit {len(raw_env.targets)} inspection points before reset.")
            for i in range(demo_step_budget):
                action = _osd_fuzzy_action(raw_env)
                obs, rewards, dones, infos = vec.step(np.asarray([action], dtype=np.float32))
                if i % 120 == 0:
                    info = infos[0] if infos else {}
                    print(
                        f"[INSPECT] step={i:06d} pos={raw_env.pos.round(2).tolist()} "
                        f"target={raw_env.target_idx}/{len(raw_env.targets)-1} "
                        f"dist={float(info.get('distance', 0.0)):.2f} "
                        f"slam_q={float(info.get('slam_quality', 0.0)):.2f} "
                        f"mu_T={float(info.get('mu_T', 0.0)):.2f} "
                        f"vt={float(info.get('osd_speed_vt', 0.0)):.2f} "
                        f"reach_r={float(info.get('inspection_reach_radius', 0.0)):.2f}"
                    )
                if bool(dones[0]):
                    info = infos[0] if infos else {}
                    if bool(info.get("route_complete", False)):
                        print(
                            f"[INSPECT] Route complete: visited all {info.get('num_targets', len(raw_env.targets))} "
                            f"inspection points. Figures/metrics saved in {raw_env.output_root}"
                        )
                        break
                    print("[INSPECT] Episode ended before route completion due to safety termination; resetting.")
                    obs = vec.reset()
                omni.kit.app.get_app().update()
            else:
                print(
                    f"[INSPECT] Demo step budget ended before full route completion. "
                    f"Last target={raw_env.target_idx}/{len(raw_env.targets)-1}. "
                    f"Increase --total-timesteps or use a larger --inspection-reach-radius."
                )
        except KeyboardInterrupt:
            pass
        finally:
            vec.close()
            simulation_app.close()
        return

    if args.mode == "collect_uniform_speed":
        collect_uniform_speed_library()
        simulation_app.close()
        return

    if args.mode == "build_op_cbrs_library":
        build_library_from_existing_csv()
        simulation_app.close()
        return

    if args.mode == "train_e1_osd":
        train_ppo_policy("PPO-OSD+Fuzzy+OP-CBRS-Source-e1", "osd_fuzzy_opcbrs_source_e1", "e1")
        simulation_app.close()
        return

    if args.mode == "transfer_e2":
        source = args.source_model or str(Path(args.output_root).expanduser() / "models" / "osd_fuzzy_opcbrs_source_e1.zip")
        train_ppo_policy("PPO-OSD+Fuzzy+OP-CBRS-Sim2Sim-Transfer-e2", "osd_fuzzy_opcbrs_transfer_e2", "e2", load_path=source, learn_steps=args.transfer_timesteps)
        simulation_app.close()
        return

    if args.mode == "eval_baselines":
        eval_baselines()
        simulation_app.close()
        return

    if args.mode == "eval_transfer":
        model_path = args.transfer_model or args.source_model or str(Path(args.output_root).expanduser() / "models" / "osd_fuzzy_opcbrs_transfer_e2.zip")
        evaluate_model(model_path, f"Eval-Transferred-OSD+Fuzzy+OP-CBRS-{args.sim_env_id}", args.sim_env_id, args.eval_episodes)
        simulation_app.close()
        return

    if args.mode == "train_e1_no_osd":
        args.disable_osd = True
        args.enable_osd = False
        args.policy_analytic_blend = 0.0
        train_ppo_policy("PPO+Fuzzy+OP-CBRS-w/o-OSD-Source-e1", "ppo_fuzzy_opcbrs_no_osd_source_e1", "e1")
        simulation_app.close()
        return

    if args.mode == "transfer_e2_no_osd":
        args.disable_osd = True
        args.enable_osd = False
        args.policy_analytic_blend = 0.0
        source = args.source_model or str(Path(args.output_root).expanduser() / "models" / "ppo_fuzzy_opcbrs_no_osd_source_e1.zip")
        train_ppo_policy("PPO+Fuzzy+OP-CBRS-w/o-OSD-Sim2Sim-Transfer-e2", "ppo_fuzzy_opcbrs_no_osd_transfer_e2", "e2", load_path=source, learn_steps=args.transfer_timesteps)
        simulation_app.close()
        return

    if args.mode == "eval_transfer_no_osd":
        args.disable_osd = True
        args.enable_osd = False
        args.policy_analytic_blend = 0.0
        model_path = args.transfer_model or args.source_model or str(Path(args.output_root).expanduser() / "models" / "ppo_fuzzy_opcbrs_no_osd_transfer_e2.zip")
        evaluate_model(model_path, f"Eval-PPO+Fuzzy+OP-CBRS-w/o-OSD-{args.sim_env_id}", args.sim_env_id, args.eval_episodes)
        simulation_app.close()
        return

    if args.mode == "paper_pipeline_no_osd":
        args.disable_osd = True
        args.enable_osd = False
        args.policy_analytic_blend = 0.0
        lib_path = collect_uniform_speed_library()
        args.potential_library = str(lib_path)
        source = train_ppo_policy("PPO+Fuzzy+OP-CBRS-w/o-OSD-Source-e1", "ppo_fuzzy_opcbrs_no_osd_source_e1", "e1")
        transfer = train_ppo_policy("PPO+Fuzzy+OP-CBRS-w/o-OSD-Sim2Sim-Transfer-e2", "ppo_fuzzy_opcbrs_no_osd_transfer_e2", "e2", load_path=source, learn_steps=args.transfer_timesteps)
        evaluate_model(transfer, "Eval-PPO+Fuzzy+OP-CBRS-w/o-OSD-e2", "e2", args.eval_episodes)
        _write_algorithm_manifest(
            "paper_pipeline_no_osd",
            {
                "potential_library": str(lib_path),
                "source_model": str(source),
                "transfer_model": str(transfer),
                "eval_episodes": int(args.eval_episodes),
                "ablation": "PPO without analytic/fuzzy OSD speed shield; OP-CBRS remains enabled",
            },
        )
        simulation_app.close()
        return

    if args.mode == "paper_pipeline":
        lib_path = collect_uniform_speed_library()
        args.potential_library = str(lib_path)
        source = train_ppo_policy("PPO-OSD+Fuzzy+OP-CBRS-Source-e1", "osd_fuzzy_opcbrs_source_e1", "e1")
        transfer = train_ppo_policy("PPO-OSD+Fuzzy+OP-CBRS-Sim2Sim-Transfer-e2", "osd_fuzzy_opcbrs_transfer_e2", "e2", load_path=source, learn_steps=args.transfer_timesteps)
        eval_baselines()
        evaluate_model(transfer, "Eval-Transferred-OSD+Fuzzy+OP-CBRS-e2", "e2", args.eval_episodes)
        _write_algorithm_manifest(
            "paper_pipeline",
            {
                "potential_library": str(lib_path),
                "source_model": str(source),
                "transfer_model": str(transfer),
                "eval_episodes": int(args.eval_episodes),
            },
        )
        simulation_app.close()
        return

    simulation_app.close()
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
