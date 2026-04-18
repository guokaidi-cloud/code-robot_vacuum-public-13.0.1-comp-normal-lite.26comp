#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。

Protocol structs (frame_state):
  - NpcState: npc_id, pos{x,z}, idx (1–4) / 官方机器人
  - OrganState: sub_type (1=充电桩), config_id, pos{x,z}, w, h (充电桩为 3×3 格)
"""

import math
from collections import deque

import numpy as np

# 8-neighborhood, same as agent move directions / 与八向移动一致
_NEI8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def _as_obj_list(raw):
    """Normalize npcs/organs to a list of dicts (handles list or id→dict map)."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [v for v in raw.values() if isinstance(v, dict)]
    if isinstance(raw, (list, tuple)):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _dist_point_to_grid_rect(px, pz, x0, z0, w, h):
    """Min Euclidean distance from integer point to w×h axis-aligned grid rectangle (inclusive)."""
    if w < 1 or h < 1:
        return 200.0
    x1 = x0 + w - 1
    z1 = z0 + h - 1
    cx = max(x0, min(int(px), x1))
    cz = max(z0, min(int(pz), z1))
    return float(math.hypot(px - cx, pz - cz))


def _point_in_grid_rect(px, pz, x0, z0, w, h):
    if w < 1 or h < 1:
        return False
    return x0 <= int(px) <= x0 + w - 1 and z0 <= int(pz) <= z0 + h - 1


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    # OrganState.sub_type: 1 = 充电桩（官方文档）
    ORGAN_SUBTYPE_CHARGER = 1

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径

    # BFS upper bound for normalization (unreachable → treat as this) / 不可达时的步数上界
    BFS_MAX_STEPS = 260
    # Slack = battery - bfs_steps: clip range for normalization
    # 电量减最短路步数：略大于典型满电，便于表达「不够赶回桩」
    SLACK_CLIP = 120.0
    # 电量相对 BFS 最短路：已进入「必须立刻去桩」区，再扫/绕路极可能到不了桩（可调）
    # margin：仍允许比最短路多耗一点点电的缓冲（转弯/探测误差）；再超就必须直线去桩
    CHARGE_URGENCY_MARGIN = 5
    # 临界：电量已不足以覆盖最短路步数（再不走基本必挂）
    CHARGE_CRITICAL_PENALTY = -0.55
    # 紧迫：电量 <= 最短路 + margin，必须立即去桩，不允许再「尽快」拖延
    CHARGE_IMMEDIATE_PENALTY = -0.35

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.npcs = []
        self.organs = []

        self.step_no = 0
        self.battery = 600
        self.battery_max = 600
        self._prev_battery = None
        self.battery_delta = 0

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # Cached for reward_process (set in _get_global_state_feature)
        self._bfs_steps_to_charger = self.BFS_MAX_STEPS
        self._on_charger_flag = False

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.npcs = _as_obj_list(frame_state.get("npcs"))
        self.organs = _as_obj_list(frame_state.get("organs"))

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        new_bat = int(hero["battery"])
        if self._prev_battery is None:
            self.battery_delta = 0
        else:
            self.battery_delta = new_bat - self._prev_battery
        self._prev_battery = new_bat
        self.battery = new_bat
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _charger_goal_cells(self):
        """All grid cells occupied by charging piles (OrganState sub_type=1)."""
        goals = set()
        for org in self.organs:
            if int(org.get("sub_type", -1)) != self.ORGAN_SUBTYPE_CHARGER:
                continue
            pos = org.get("pos") or {}
            x0 = int(pos.get("x", 0))
            z0 = int(pos.get("z", 0))
            w = max(int(org.get("w", 3)), 1)
            h = max(int(org.get("h", 3)), 1)
            for dx in range(w):
                for dz in range(h):
                    gx, gz = x0 + dx, z0 + dz
                    if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                        goals.add((gx, gz))
        return goals

    def _bfs_min_steps_to_goals(self, sx, sz, goals):
        """8-connected BFS on passable_map; min steps to enter any goal cell.

        在已知可通行格子上做 BFS（与八向一步一格一致），到达任一目标格的最少步数。
        未探索区域在 passable_map 中仍为 1，可能偏乐观；障碍已观测为 0。
        """
        if not goals:
            return self.BFS_MAX_STEPS
        if (sx, sz) in goals:
            return 0
        q = deque()
        q.append((sx, sz, 0))
        visited = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.bool_)
        visited[sx, sz] = True
        while q:
            x, z, d = q.popleft()
            for dx, dz in _NEI8:
                nx, nz = x + dx, z + dz
                if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
                    continue
                if visited[nx, nz]:
                    continue
                if self.passable_map[nx, nz] == 0:
                    continue
                if (nx, nz) in goals:
                    return d + 1
                visited[nx, nz] = True
                q.append((nx, nz, d + 1))
        return self.BFS_MAX_STEPS

    def _charger_features_from_organs(self):
        """Use OrganState (sub_type=1, pos, w, h) for global charger geometry.

        使用 OrganState：充电桩为 3×3（默认 w=h=3），计算到桩区距离与是否站在桩上。
        """
        hx, hz = self.cur_pos
        best = 200.0
        on = False
        for org in self.organs:
            if int(org.get("sub_type", -1)) != self.ORGAN_SUBTYPE_CHARGER:
                continue
            pos = org.get("pos") or {}
            x0 = int(pos.get("x", 0))
            z0 = int(pos.get("z", 0))
            w = max(int(org.get("w", 3)), 1)
            h = max(int(org.get("h", 3)), 1)
            d = _dist_point_to_grid_rect(hx, hz, x0, z0, w, h)
            if d < best:
                best = d
            if _point_in_grid_rect(hx, hz, x0, z0, w, h):
                on = True
        return best, on

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21.

        局部视野特征（49D）：从 21×21 视野中心裁剪 7×7。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        """Global state feature (16D).

        全局状态特征（16D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
          [7]  ray_E_dirt        east ray distance / 向右（x+）方向
          [8]  ray_S_dirt        south ray distance / 向下（z+）方向
          [9]  ray_W_dirt        west ray distance / 向左（x-）方向
          [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
          [12] nearest_charger_norm 直线到桩区几何（欧氏）/ 启发用
          [13] on_charger        是否站在任一充电桩 3×3 区域内（1=是, 0=否）
          [14] bfs_charger_norm  沿可通行格八连通的 BFS 最少步数（归一化），近似「赶回桩所需步数」
          [15] battery_bfs_slack_norm  (当前电量 − BFS 步数) 归一化；明显为负表示很难赶回桩
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(x - (hx - self.VIEW_HALF), 0, 20), np.clip(z - (hz - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        nearest_charger_dist, on_charger_flag = self._charger_features_from_organs()
        nearest_charger_norm = _norm(nearest_charger_dist, 180.0)
        on_charger = 1.0 if on_charger_flag else 0.0

        goals = self._charger_goal_cells()
        bfs_steps = self._bfs_min_steps_to_goals(hx, hz, goals)
        self._bfs_steps_to_charger = int(bfs_steps)
        self._on_charger_flag = bool(on_charger_flag)
        bfs_charger_norm = _norm(float(min(bfs_steps, self.BFS_MAX_STEPS)), float(self.BFS_MAX_STEPS))
        slack = float(self.battery) - float(bfs_steps)
        slack_c = float(np.clip(slack, -self.SLACK_CLIP, self.SLACK_CLIP))
        battery_bfs_slack_norm = _norm(slack_c, self.SLACK_CLIP, -self.SLACK_CLIP)

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
                nearest_charger_norm,
                on_charger,
                bfs_charger_norm,
                battery_bfs_slack_norm,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)

    def feature_process(self, env_obs, last_action):
        """Generate 73D feature vector, legal action mask, and scalar reward.

        生成 73D 特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()  # 16D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr])  # 73D

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        # Cleaning reward / 清扫奖励
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_reward = 0.1 * cleaned_this_step

        # Charging reward / 在桩上回充时电量上升
        charge_reward = 0.05 * max(0, self.battery_delta)

        br = self.battery / float(self.battery_max)
        low_battery_penalty = -0.002 if br < 0.25 else 0.0

        # 相对 BFS 最短路：必须「立即」去桩（已在桩上则不罚）。不是「尽快」，拖延会到不了桩。
        # 仅当 BFS 为有限最短路时启用（不可达占位不触发）
        urgency_penalty = 0.0
        if (
            self._charger_goal_cells()
            and not self._on_charger_flag
            and self._bfs_steps_to_charger < self.BFS_MAX_STEPS
        ):
            b = int(self.battery)
            s = int(self._bfs_steps_to_charger)
            m = int(self.CHARGE_URGENCY_MARGIN)
            if b <= s:
                urgency_penalty = float(self.CHARGE_CRITICAL_PENALTY)
            elif b <= s + m:
                urgency_penalty = float(self.CHARGE_IMMEDIATE_PENALTY)

        # Step penalty / 时间惩罚
        step_penalty = -0.001

        return cleaning_reward + charge_reward + low_battery_penalty + urgency_penalty + step_penalty
