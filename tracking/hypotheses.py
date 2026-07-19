"""Adaptive generation and de-duplication of local/global pose candidates."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from tracking.config import TrackerConfig
from tracking.flow import FlowMeasurement, SparseFlowPoseUpdater
from tracking.geometry import (
    flipped_pose,
    project_center,
    rotate_pose,
    rotation_error_deg,
    propagate_pose_window,
    shift_projected_center,
    translate_pose,
    translation_error_m,
)
from tracking.types import Detection, FrameData, PoseHypothesis, Track, TrackMode


class AdaptiveHypothesisGenerator:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.flow_updater = SparseFlowPoseUpdater(config.flow)

    def propagated_pose(self, track: Track) -> np.ndarray:
        if len(track.history) >= 2:
            return propagate_pose_window(
                track.history,
                self.config.temporal.window_size,
                self.config.temporal.robust_velocity,
            )
        return track.propagated_pose()

    def generate(
        self,
        *,
        frame: FrameData,
        detection: Detection,
        track: Track | None,
        fresh: list[PoseHypothesis],
        mode: TrackMode,
        periodic_global: bool,
        flow_measurement: FlowMeasurement | None,
    ) -> list[PoseHypothesis]:
        candidates: list[PoseHypothesis] = []
        propagated: PoseHypothesis | None = None
        if track is not None:
            propagated = PoseHypothesis(
                pose=self.propagated_pose(track),
                source="constant_velocity",
                measurement_score=track.confidence,
                obj_id=track.obj_id,
            )
            flow_candidate = None
            if flow_measurement is not None:
                flow_candidate = PoseHypothesis(
                    pose=self.flow_updater.update_pose(
                        track.pose, frame.K, flow_measurement
                    ),
                    source="optical_flow",
                    measurement_score=flow_measurement.confidence,
                    obj_id=track.obj_id,
                )
            if mode == TrackMode.NORMAL:
                # Cheap path: exactly one temporal proposal before local refinement.
                candidates.append(flow_candidate or propagated)
            else:
                candidates.extend(
                    [
                        propagated,
                        PoseHypothesis(
                            pose=track.pose.copy(),
                            source="previous_pose",
                            measurement_score=track.confidence,
                            obj_id=track.obj_id,
                        ),
                    ]
                )
                if flow_candidate is not None:
                    candidates.append(flow_candidate)

        if mode == TrackMode.LOST or track is None:
            count = self.config.hypotheses.top_k_relocalization
        elif mode == TrackMode.UNCERTAIN:
            count = self.config.hypotheses.top_k_uncertain
        else:
            count = 1 if periodic_global else 0
        candidates.extend(
            item.copy(source=f"{item.source}_global_rank_{rank}")
            for rank, item in enumerate(fresh[:count])
        )

        if mode in {TrackMode.UNCERTAIN, TrackMode.LOST}:
            seeds = list(candidates)
            if self.config.hypotheses.add_flip_when_uncertain:
                for seed in seeds:
                    candidates.append(
                        PoseHypothesis(
                            pose=flipped_pose(
                                seed.pose, self.config.hypotheses.flip_axis
                            ),
                            source=f"{seed.source}_flip180",
                            measurement_score=seed.measurement_score * 0.8,
                            obj_id=seed.obj_id,
                        )
                    )
            # Broad translation-space perturbations are only used while lost.
            if mode == TrackMode.LOST:
                for seed in seeds[: max(1, min(2, len(seeds)))]:
                    for offset in self.config.hypotheses.translation_offsets_m:
                        for axis in (0, 1, 2):
                            delta = np.zeros(3)
                            delta[axis] = float(offset)
                            candidates.append(
                                PoseHypothesis(
                                    pose=translate_pose(seed.pose, delta),
                                    source=f"{seed.source}_t{axis}_{offset:+.3f}",
                                    measurement_score=seed.measurement_score * 0.75,
                                    obj_id=seed.obj_id,
                                )
                            )

        if mode != TrackMode.NORMAL:
            seeds = list(candidates[: max(1, min(3, len(candidates)))])
            for seed in seeds:
                projected_center, valid_center = project_center(seed.pose, frame.K)
                if valid_center:
                    candidates.append(
                        PoseHypothesis(
                            pose=shift_projected_center(
                                seed.pose,
                                frame.K,
                                delta_uv_px=(
                                    np.asarray(detection.center) - projected_center
                                ),
                            ),
                            source=f"{seed.source}_align_detection_center",
                            measurement_score=seed.measurement_score * 0.9,
                            obj_id=seed.obj_id,
                        )
                    )
                for center_offset in self.config.hypotheses.center_offsets_px:
                    for axis in (0, 1):
                        delta_uv = np.zeros(2)
                        delta_uv[axis] = float(center_offset)
                        candidates.append(
                            PoseHypothesis(
                                pose=shift_projected_center(
                                    seed.pose, frame.K, delta_uv_px=delta_uv
                                ),
                                source=f"{seed.source}_uv{axis}_{center_offset:+.0f}",
                                measurement_score=seed.measurement_score * 0.85,
                                obj_id=seed.obj_id,
                            )
                        )
                for log_depth in self.config.hypotheses.log_depth_offsets:
                    candidates.append(
                        PoseHypothesis(
                            pose=shift_projected_center(
                                seed.pose,
                                frame.K,
                                delta_log_depth=float(log_depth),
                            ),
                            source=f"{seed.source}_logz_{log_depth:+.3f}",
                            measurement_score=seed.measurement_score * 0.85,
                            obj_id=seed.obj_id,
                        )
                    )
                for yaw_deg in self.config.hypotheses.yaw_offsets_deg:
                    candidates.append(
                        PoseHypothesis(
                            pose=rotate_pose(
                                seed.pose,
                                np.asarray([0.0, 0.0, math.radians(yaw_deg)]),
                                side="left",
                            ),
                            source=f"{seed.source}_yaw_{yaw_deg:+.1f}",
                            measurement_score=seed.measurement_score * 0.85,
                            obj_id=seed.obj_id,
                        )
                    )
        return self._deduplicate(candidates)

    def _deduplicate(
        self, candidates: Iterable[PoseHypothesis]
    ) -> list[PoseHypothesis]:
        kept: list[PoseHypothesis] = []
        for candidate in candidates:
            if candidate.pose[2, 3] <= 1e-5 or not np.isfinite(candidate.pose).all():
                continue
            duplicate = any(
                translation_error_m(candidate.pose, existing.pose)
                <= self.config.hypotheses.deduplicate_translation_m
                and rotation_error_deg(candidate.pose, existing.pose)
                <= self.config.hypotheses.deduplicate_rotation_deg
                for existing in kept
            )
            if not duplicate:
                kept.append(candidate)
        return kept
