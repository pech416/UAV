"""vectorized_ops.py

Vectorized operations for the UAV project.

Fix:
- `VectorizedOps.find_best_uav_for_tasks` accepts the keyword argument `timing`.

Speed (CPU):
- Task assignment removes the inner per-UAV Python loop by using:
  * precomputed cache masks (content / service-only / neither)
  * NumPy vectorized compute-time evaluation
  * identical random-number consumption order for the service-cache branch

Notes on determinism:
- For a given NumPy RNG state, the service branch uses
  `np.random.randint(3, 8, size=n_service_hits)` in ascending UAV index order,
  which is equivalent to drawing one randint per service-hit UAV inside a
  `for uav_idx in range(n_uav)` loop.
"""

import math
import time
from typing import Dict, List, Tuple, Optional

import numpy as np


class VectorizedOps:
    """Vectorized operations for UAV task processing."""

    # -------------------- task-assign cache (rebuilt per call, buffers reused) --------------------
    _ta_T: Optional[int] = None
    _ta_n_uav: Optional[int] = None

    _ta_content_mask: Optional[np.ndarray] = None        # (T, n_uav) bool
    _ta_service_only_mask: Optional[np.ndarray] = None   # (T, n_uav) bool  (service & ~content)
    _ta_neither_mask: Optional[np.ndarray] = None        # (T, n_uav) bool
    _ta_service_idxs: Optional[List[np.ndarray]] = None  # list[T] of int arrays

    _ta_F_cpu: Optional[np.ndarray] = None               # (n_uav,) float64
    _ta_inv_uv_rate: Optional[np.ndarray] = None         # (n_uav,) float64

    # reusable buffers
    _ta_base: Optional[np.ndarray] = None                # (n_uav,) float64
    _ta_term_user: Optional[np.ndarray] = None           # (n_uav,) float64
    _ta_term_uv: Optional[np.ndarray] = None             # (n_uav,) float64
    _ta_times: Optional[np.ndarray] = None               # (n_uav,) float64

    @staticmethod
    def _prepare_task_assign_cache(uavs: List, task_type_number: int = 400) -> None:
        """Build cache masks for current UAV cache state.

        Always rebuild values from current uav.cached_* lists to guarantee correctness.
        Arrays are reused when shapes match to reduce allocations.
        """
        n_uav = len(uavs)
        T = int(task_type_number) + 1

        need_alloc = (
            VectorizedOps._ta_content_mask is None
            or VectorizedOps._ta_content_mask.shape != (T, n_uav)
        )

        if need_alloc:
            VectorizedOps._ta_T = T
            VectorizedOps._ta_n_uav = n_uav
            VectorizedOps._ta_content_mask = np.zeros((T, n_uav), dtype=np.bool_)
            VectorizedOps._ta_service_only_mask = np.zeros((T, n_uav), dtype=np.bool_)
            VectorizedOps._ta_neither_mask = np.zeros((T, n_uav), dtype=np.bool_)
            VectorizedOps._ta_service_idxs = [np.empty((0,), dtype=np.int32) for _ in range(T)]

            VectorizedOps._ta_F_cpu = np.empty((n_uav,), dtype=np.float64)
            VectorizedOps._ta_inv_uv_rate = np.empty((n_uav,), dtype=np.float64)

            VectorizedOps._ta_base = np.empty((n_uav,), dtype=np.float64)
            VectorizedOps._ta_term_user = np.empty((n_uav,), dtype=np.float64)
            VectorizedOps._ta_term_uv = np.empty((n_uav,), dtype=np.float64)
            VectorizedOps._ta_times = np.empty((n_uav,), dtype=np.float64)
        else:
            # reuse arrays
            VectorizedOps._ta_T = T
            VectorizedOps._ta_n_uav = n_uav
            VectorizedOps._ta_content_mask.fill(False)
            VectorizedOps._ta_service_only_mask.fill(False)

        content_mask = VectorizedOps._ta_content_mask
        service_only = VectorizedOps._ta_service_only_mask

        F_cpu = VectorizedOps._ta_F_cpu
        inv_uv_rate = VectorizedOps._ta_inv_uv_rate

        # Fill masks per UAV
        for i, u in enumerate(uavs):
            # content
            try:
                ct = u.cached_content_type
            except Exception:
                ct = []
            if ct:
                arr = np.asarray(ct, dtype=np.int32)
                # guard range
                arr = arr[(arr >= 0) & (arr < T)]
                if arr.size:
                    content_mask[arr, i] = True

            # service
            try:
                st = u.cached_service_type
            except Exception:
                st = []
            if st:
                arr = np.asarray(st, dtype=np.int32)
                arr = arr[(arr >= 0) & (arr < T)]
                if arr.size:
                    service_only[arr, i] = True

            # parameters
            F_cpu[i] = float(getattr(u, "F_cpu", 0.0))
            inv_uv_rate[i] = 1.0 / max(float(getattr(u, "U_V_transmission_rate", 0.0)), 1e-12)

        # service_only = service & ~content (vectorized)
        np.logical_and(service_only, ~content_mask, out=service_only)

        # neither = ~(content | service_only)
        union = np.logical_or(content_mask, service_only)
        np.logical_not(union, out=VectorizedOps._ta_neither_mask)

        # precompute indices for each task type (ascending UAV index order)
        VectorizedOps._ta_service_idxs = [np.flatnonzero(service_only[t]).astype(np.int32, copy=False) for t in range(T)]

    # -------------------- distance & transmission --------------------
    @staticmethod
    def calculate_distances_batch(uav_positions: np.ndarray, user_positions: np.ndarray, cloud_position) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized distances.

        uav_positions: (n_uavs, 3) [x, y, z]
        user_positions: (n_users, 2) [x, y]
        cloud_position: [x, y]
        """
        uav_positions = np.asarray(uav_positions, dtype=np.float32)
        user_positions = np.asarray(user_positions, dtype=np.float32)

        uav_xy = uav_positions[:, :2]
        uav_z = uav_positions[:, 2]

        dx = uav_xy[:, 0:1] - user_positions[:, 0]
        dy = uav_xy[:, 1:2] - user_positions[:, 1]
        d_level = np.sqrt(dx * dx + dy * dy)
        distances_uav_user = np.sqrt(d_level * d_level + (uav_z[:, None] * uav_z[:, None]))

        cloud_xy = np.asarray(cloud_position, dtype=np.float32)
        dx_cloud = uav_xy[:, 0] - cloud_xy[0]
        dy_cloud = uav_xy[:, 1] - cloud_xy[1]
        distances_uav_cloud = np.sqrt(dx_cloud * dx_cloud + dy_cloud * dy_cloud)

        return distances_uav_user, distances_uav_cloud

    @staticmethod
    def calculate_transmission_rates_batch(distances_uav_user: np.ndarray, distances_uav_cloud: np.ndarray, uav_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized Shannon rates."""
        distances_uav_user = np.asarray(distances_uav_user, dtype=np.float64)
        distances_uav_cloud = np.asarray(distances_uav_cloud, dtype=np.float64)

        # UAV -> cloud
        uav_to_cloud_rates = (
            10.0 * float(uav_params['B_cloud'])
            * np.log2(
                1.0
                + 10000.0
                * float(uav_params['P_u'])
                * np.power(distances_uav_cloud / 100.0, -float(uav_params['r']))
                / (float(uav_params['sigma2']) + float(uav_params['N']))
            )
        )

        # User -> UAV
        user_to_uav_rates = (
            100.0 * float(uav_params['B'])
            * np.log2(
                1.0
                + 5000.0
                * (
                    float(uav_params['P_v_w'])
                    * np.power(distances_uav_user / 10.0, -float(uav_params['r']))
                    / (float(uav_params['sigma2']) + float(uav_params['N']))
                )
            )
        )

        return user_to_uav_rates, uav_to_cloud_rates

    # -------------------- task assignment (timing aware) --------------------
    @staticmethod
    def find_best_uav_for_tasks(tasks, uavs: List, users: List, cloud_position, timing=None):
        """Return {(user_idx, task_idx): (best_uav_idx, min_time)}.

        `timing` is optional. When provided, it should support:
            timing.add(key: str, seconds: float)
        and will be updated for: 'distance', 'trans_rate', 'cache_check', 'comp_time'.
        """
        if not uavs or not users:
            return {}

        task_type_number = getattr(users[0], 'task_type_number', 400)
        VectorizedOps._prepare_task_assign_cache(uavs, task_type_number=task_type_number)

        # --- distances ---
        if timing is not None:
            t0 = time.perf_counter()
        uav_positions = np.asarray([u.position for u in uavs], dtype=np.float32)
        user_positions = np.asarray([u.position for u in users], dtype=np.float32)
        distances_uav_user, distances_uav_cloud = VectorizedOps.calculate_distances_batch(
            uav_positions, user_positions, cloud_position
        )
        if timing is not None:
            timing.add('distance', time.perf_counter() - t0)

        # --- transmission rates ---
        if timing is not None:
            t0 = time.perf_counter()
        u0 = uavs[0]
        uav_params = {
            'P_u': u0.P_u,
            'P_v_w': u0.P_v_w,
            'r': u0.r,
            'sigma2': u0.sigma2,
            'N': u0.N,
            'B': u0.B,
            'B_cloud': u0.B_cloud,
        }
        r_user, r_cloud = VectorizedOps.calculate_transmission_rates_batch(
            distances_uav_user, distances_uav_cloud, uav_params
        )
        if timing is not None:
            timing.add('trans_rate', time.perf_counter() - t0)

        inv_r_cloud = 1.0 / np.maximum(np.asarray(r_cloud, dtype=np.float64), 1e-6)

        # reusable buffers
        n_uav = VectorizedOps._ta_n_uav
        base = VectorizedOps._ta_base
        term_user = VectorizedOps._ta_term_user
        term_uv = VectorizedOps._ta_term_uv
        times = VectorizedOps._ta_times

        assign = {}

        # NOTE: we intentionally loop users/tasks in Python (usually far fewer than uav loop)
        # while keeping per-task per-uav computation vectorized.
        for ui, user in enumerate(users):
            inv_r_user = 1.0 / np.maximum(np.asarray(r_user[:, ui], dtype=np.float64), 1e-6)

            for ti, task in enumerate(user.tasks):
                # task[-1] marks completed task
                if task[-1]:
                    continue

                ttype = int(task[0])
                tsize = float(task[1])

                # cache check (very cheap now, but kept for your timing breakdown)
                if timing is not None:
                    tc0 = time.perf_counter()
                if 0 <= ttype < VectorizedOps._ta_T:
                    service_idxs = VectorizedOps._ta_service_idxs[ttype]
                    neither_mask = VectorizedOps._ta_neither_mask[ttype]
                else:
                    service_idxs = np.empty((0,), dtype=np.int32)
                    neither_mask = np.ones((n_uav,), dtype=np.bool_)
                if timing is not None:
                    timing.add('cache_check', time.perf_counter() - tc0)

                # comp time
                if timing is not None:
                    ts0 = time.perf_counter()

                base.fill(0.0)  # content-hit stays 0

                # service-only hit: consume RNG in ascending UAV index order (matches scalar loop)
                if service_idxs.size:
                    rnd = np.random.randint(3, 8, size=service_idxs.size)
                    computing_capacity = rnd * VectorizedOps._ta_F_cpu[service_idxs] / 30.0
                    base[service_idxs] = tsize / computing_capacity

                # neither: cloud branch
                if neither_mask.any():
                    base[neither_mask] = 2.0 * tsize * inv_r_cloud[neither_mask]

                # user uplink
                np.multiply(inv_r_user, tsize, out=term_user)
                np.add(base, term_user, out=base)

                # UAV internal transmission
                np.multiply(VectorizedOps._ta_inv_uv_rate, tsize * 0.6, out=term_uv)
                np.add(base, term_uv, out=times)

                best_uav = int(np.argmin(times))
                best_time = float(times[best_uav])
                assign[(ui, ti)] = (best_uav, best_time)

                if timing is not None:
                    timing.add('comp_time', time.perf_counter() - ts0)

        return assign

    # -------------------- other helpers (unchanged) --------------------
    @staticmethod
    def move_uavs_batch(uav_positions: np.ndarray, actions: np.ndarray, move_distance: float = 10.0) -> np.ndarray:
        """Move all UAVs in a vectorized manner."""
        move_directions = np.pi * actions + np.pi
        dx = move_distance * np.cos(move_directions)
        dy = move_distance * np.sin(move_directions)
        new_positions = np.array(uav_positions, copy=True)
        new_positions[:, 0] += dx
        new_positions[:, 1] += dy
        return new_positions

    @staticmethod
    def enforce_min_distance_batch(uav_positions: np.ndarray, min_distance: float = 400.0, max_range: float = 20000.0) -> np.ndarray:
        """Enforce minimum distance constraint between UAVs."""
        n_uavs = uav_positions.shape[0]
        positions = np.array(uav_positions, copy=True)
        for i in range(n_uavs):
            for j in range(i + 1, n_uavs):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < min_distance and distance > 1e-12:
                    overlap = min_distance - distance
                    positions[i, 0] += overlap * dx / 2.0
                    positions[i, 1] += overlap * dy / 2.0
                    positions[j, 0] -= overlap * dx / 2.0
                    positions[j, 1] -= overlap * dy / 2.0
        positions[:, 0] = np.clip(positions[:, 0], -max_range, max_range)
        positions[:, 1] = np.clip(positions[:, 1], -max_range, max_range)
        return positions

    @staticmethod
    def calculate_uav_rewards_batch(uav_profits: np.ndarray, uav_base_costs: np.ndarray, uav_tasked_numbers: np.ndarray, task_num_sills: np.ndarray):
        """Calculate rewards for all UAVs in a vectorized manner."""
        rewards = uav_profits - uav_base_costs
        task_excess = np.maximum(uav_tasked_numbers - task_num_sills, 0)
        updated_base_costs = uav_base_costs + task_excess * 3
        updated_sills = task_num_sills + task_excess
        return rewards, updated_base_costs, updated_sills
