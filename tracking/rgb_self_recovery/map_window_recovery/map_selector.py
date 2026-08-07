"""Soft track-map diagnostics, reranking, and Frenet-window motion costs."""

from __future__ import annotations

from dataclasses import replace
import math
import numpy as np

from tracking.rgb_self_recovery.sliding_window.optimizer import SlidingWindowSelector

from .track_map import TrackMap
from .transforms import MapTransformProvider


def _rho(value):
    return float(2 * (math.sqrt(1 + float(value) ** 2) - 1))


class MapWindowSelector(SlidingWindowSelector):
    def __init__(self, args):
        super().__init__(args)
        self.track_map = TrackMap(args.track_map) if args.track_map else None
        self.transforms = MapTransformProvider(args)
        self.map_weight = float(args.map_weight)
        self.motion_weight = float(args.map_motion_weight)
        self.boundary_sigma = float(args.map_boundary_sigma_m)
        self.height_sigma = float(args.map_height_sigma_m)
        self.expected_height = float(args.map_expected_center_height_m)
        self.heading_sigma = math.radians(float(args.map_heading_sigma_deg))
        self.half_width = .5 * float(args.map_car_width_m)
        self.axis = {"x": 0, "y": 1, "z": 2}[args.map_forward_axis]
        self.allow_reverse = bool(args.map_allow_reverse)
        self.speed_sigma_s = float(args.map_speed_sigma_mps)
        self.accel_sigma_s = float(args.map_acceleration_sigma_mps2)
        self.lateral_sigma = float(args.map_lateral_speed_sigma_mps)
        self.backward_sigma = float(args.map_backward_speed_sigma_mps)
        self.hard_boundary = float(args.map_hard_boundary_violation_m)
        self._map = {}
        self.available = self.missing = 0
        for name, value in (
            ("map boundary sigma", self.boundary_sigma),
            ("map height sigma", self.height_sigma),
            ("map heading sigma", self.heading_sigma),
            ("map speed sigma", self.speed_sigma_s),
            ("map acceleration sigma", self.accel_sigma_s),
            ("map lateral speed sigma", self.lateral_sigma),
            ("map backward speed sigma", self.backward_sigma),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def report(self):
        return {
            "map_enabled": self.track_map is not None,
            "map_weight": self.map_weight,
            "map_motion_weight": self.motion_weight,
            "map_candidates_available": self.available,
            "map_candidates_missing_transform": self.missing,
        }

    def _score(self, pose_camera, map_camera, reference_s=None):
        pose_map = map_camera @ np.asarray(pose_camera, dtype=float)
        projection = self.track_map.project(pose_map[:3, 3], reference_s)
        d = projection["d"]
        violation = max(
            0., d + self.half_width - projection["right_width"],
            -projection["left_width"] - (d - self.half_width),
        )
        forward = pose_map[:3, self.axis]
        forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
        tangent = projection["tangent"]
        cosine = float(np.clip(forward @ tangent, -1, 1))
        heading = math.acos(cosine)
        if self.allow_reverse:
            heading = min(heading, math.pi - heading)
        boundary_cost = _rho(violation / self.boundary_sigma)
        height_cost = _rho(
            abs(projection["h"] - self.expected_height) / self.height_sigma
        )
        heading_cost = _rho(heading / self.heading_sigma)
        return {
            **projection, "boundary_violation_m": violation,
            "heading_error_deg": math.degrees(heading),
            "map_boundary_cost": boundary_cost, "map_height_cost": height_cost,
            "map_heading_cost": heading_cost,
            "map_unary_cost": boundary_cost + height_cost + heading_cost,
        }

    def _sequence_cost(self, frames, indices):
        total, parts = super()._sequence_cost(frames, indices)
        states = [self._map.get(id(frame.candidates[index])) for frame, index in zip(frames, indices)]
        if not states or any(value is None for value in states):
            return total, parts
        times = np.asarray([frame.time_s for frame in frames])
        s_velocity, d_velocity = [], []
        backward = speed_consistency = acceleration = lateral = 0.
        for i in range(1, len(states)):
            dt = max(times[i] - times[i - 1], 1 / self.fps)
            vs = self.track_map.signed_delta(states[i]["s"], states[i - 1]["s"]) / dt
            vd = (states[i]["d"] - states[i - 1]["d"]) / dt
            s_velocity.append(vs)
            d_velocity.append(vd)
            lateral += _rho(abs(vd) / self.lateral_sigma)
            if not self.allow_reverse:
                backward += _rho(max(0., -vs) / self.backward_sigma)
        for i in range(1, len(s_velocity)):
            dt = max(times[i + 1] - times[i], 1 / self.fps)
            acceleration += _rho(abs(s_velocity[i] - s_velocity[i - 1]) / dt / self.accel_sigma_s)
        if len(s_velocity) > 1:
            median_speed = float(np.median(s_velocity))
            speed_consistency = sum(
                _rho(abs(value - median_speed) / self.speed_sigma_s)
                for value in s_velocity
            )
        motion = backward + speed_consistency + acceleration + lateral
        total += self.motion_weight * motion
        parts.update(
            map_backward_cost=backward,
            map_speed_consistency_cost=speed_consistency,
            map_acceleration_cost=acceleration,
            map_lateral_motion_cost=lateral, map_motion_cost=motion,
        )
        return total, parts

    def select(self, track_id, frame, candidates):
        if self.track_map is None:
            selected, ordered, diagnostics = super().select(track_id, frame, candidates)
            diagnostics.update(map_available=0, map_status="disabled")
            return selected, ordered, diagnostics
        map_camera, status = self.transforms.map_camera(frame)
        if map_camera is None:
            self.missing += len(candidates)
            selected, ordered, diagnostics = super().select(track_id, frame, candidates)
            diagnostics.update(map_available=0, map_status=status)
            return selected, ordered, diagnostics
        scored = []
        values = []
        for candidate in candidates:
            value = self._score(candidate.pose, map_camera)
            values.append(value)
            self.available += 1
            scored.append(replace(
                candidate,
                total_error=float(candidate.total_error + self.map_weight * value["map_unary_cost"]),
            ))
        eligible = [
            (candidate, value) for candidate, value in zip(scored, values)
            if value["boundary_violation_m"] <= self.hard_boundary
        ]
        if eligible:
            scored = [item[0] for item in eligible]
            values = [item[1] for item in eligible]
        for candidate, value in zip(scored, values):
            self._map[id(candidate)] = value
        selected, ordered, diagnostics = super().select(track_id, frame, scored)
        value = self._map.get(id(next(item for item in scored if item.source in selected.source)))
        if value is None:
            value = min(values, key=lambda item:item["map_unary_cost"])
        diagnostics.update(
            map_available=1, map_status="available", track_s_m=value["s"],
            track_d_m=value["d"], track_h_m=value["h"],
            map_boundary_violation_m=value["boundary_violation_m"],
            map_heading_error_deg=value["heading_error_deg"],
            map_unary_cost=value["map_unary_cost"],
        )
        return selected, ordered, diagnostics
