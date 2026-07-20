"""Adaptive multi-object, multi-hypothesis 6D pose tracking state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tracking.association import (
    associate_tracks,
    assign_prediction_groups,
    detection_appearance_descriptor,
)
from tracking.config import TrackerConfig
from tracking.flow import SparseFlowPoseUpdater
from tracking.geometry import rotation_error_deg, translation_error_m
from tracking.hypotheses import AdaptiveHypothesisGenerator
from tracking.recovery import RecoveryPredictor
from tracking.refinement import CandidateRefiner
from tracking.types import (
    Detection,
    EvaluatedHypothesis,
    FrameData,
    PoseHypothesis,
    Track,
    TrackMode,
)


@dataclass
class TrackedInstanceResult:
    track: Track
    detection: Detection
    chosen: EvaluatedHypothesis
    alternatives: list[EvaluatedHypothesis]
    periodic_global: bool
    elapsed_s: float
    same_frame_recovery: bool = False
    same_frame_recovery_mode: str = ""
    occluders_used: int = 0


@dataclass
class TrackingFrameResult:
    scene_id: int
    im_id: int
    instances: list[TrackedInstanceResult] = field(default_factory=list)
    unmatched_detection_ids: list[int] = field(default_factory=list)
    deleted_track_ids: list[int] = field(default_factory=list)


class AdaptivePoseTracker:
    """Confidence-gated cascade with periodic global re-detection.

    A motion prior may propose candidates, but visual CAD evidence always
    selects the final pose. This is what permits recovery after an initially
    wrong pose instead of merely propagating the error.
    """

    def __init__(
        self,
        config: TrackerConfig,
        refiner: CandidateRefiner,
        *,
        recovery: RecoveryPredictor | None = None,
    ):
        config.validate()
        self.config = config
        self.refiner = refiner
        self.recovery = recovery
        self.generator = AdaptiveHypothesisGenerator(config)
        self.flow = SparseFlowPoseUpdater(config.flow)
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 0
        self.current_scene_id: int | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 0
        self.current_scene_id = None

    def needs_global_measurement(self, detection_count: int | None = None) -> bool:
        """Tell a live caller whether full GigaPose/EPnP should run this frame."""

        active = list(self.tracks.values())
        if not active:
            return True
        if detection_count is not None and detection_count > len(active):
            return True
        return any(
            track.mode != TrackMode.NORMAL
            or track.missed > 0
            or track.age % self.config.state.safety_interval == 0
            or track.frames_since_global
            >= self.config.state.max_track_age_without_global
            for track in active
        )

    def _periodic_global(self, track: Track | None) -> bool:
        if track is None:
            return True
        return (
            track.age % self.config.state.safety_interval == 0
            or track.frames_since_global >= self.config.state.max_track_age_without_global
        )

    def _mode_for(self, track: Track | None) -> TrackMode:
        if track is None or track.missed >= self.config.state.lost_after_misses:
            return TrackMode.LOST
        return track.mode

    def _confidence_with_margin(
        self, ranked: list[EvaluatedHypothesis]
    ) -> float:
        confidence = ranked[0].score.confidence
        if len(ranked) > 1:
            margin = ranked[1].score.total_error - ranked[0].score.total_error
            reliability = float(
                np.clip(
                    margin / max(self.config.state.min_score_margin, 1e-6),
                    0.0,
                    1.0,
                )
            )
            confidence *= 0.75 + 0.25 * reliability
        return float(np.clip(confidence, 0.0, 1.0))

    def _next_mode(self, confidence: float) -> TrackMode:
        if confidence >= self.config.state.high_confidence:
            return TrackMode.NORMAL
        if confidence >= self.config.state.low_confidence:
            return TrackMode.UNCERTAIN
        return TrackMode.LOST

    def _flow_measurement(
        self, track: Track | None, frame: FrameData
    ):
        if (
            track is None
            or track.previous_gray is None
            or track.previous_mask is None
        ):
            return None
        return self.flow.estimate(
            track.previous_gray, frame.gray, track.previous_mask
        )

    def _merge_ranked(
        self,
        *ranked_groups: list[EvaluatedHypothesis],
    ) -> list[EvaluatedHypothesis]:
        """Merge search stages without duplicating equivalent poses."""

        merged: list[EvaluatedHypothesis] = []
        for evaluated in sorted(
            [item for group in ranked_groups for item in group],
            key=lambda item: item.score.total_error,
        ):
            duplicate = any(
                translation_error_m(
                    evaluated.hypothesis.pose, existing.hypothesis.pose
                )
                <= self.config.hypotheses.deduplicate_translation_m
                and rotation_error_deg(
                    evaluated.hypothesis.pose, existing.hypothesis.pose
                )
                <= self.config.hypotheses.deduplicate_rotation_deg
                for existing in merged
            )
            if not duplicate:
                merged.append(evaluated)
        return merged

    def _search(
        self,
        *,
        frame: FrameData,
        detection: Detection,
        track: Track | None,
        fresh: list[PoseHypothesis],
        mode: TrackMode,
        periodic_global: bool,
        flow_measurement,
        motion_reference: np.ndarray | None,
        occluder_poses: tuple[np.ndarray, ...],
    ) -> list[EvaluatedHypothesis]:
        candidates = self.generator.generate(
            frame=frame,
            detection=detection,
            track=track,
            fresh=fresh,
            mode=mode,
            periodic_global=periodic_global,
            flow_measurement=flow_measurement,
        )
        if not candidates:
            return []
        ranked = self.refiner.refine(
            candidates,
            frame,
            detection,
            mode,
            motion_reference,
            occluder_poses,
        )
        if self.recovery is not None and mode != TrackMode.NORMAL and ranked:
            recovery_outputs = [
                self.recovery.correct(evaluated)
                for evaluated in ranked[: min(5, len(ranked))]
            ]
            recovery_outputs.sort(key=lambda item: item[2])
            learned_candidates = [
                item[0]
                for item in recovery_outputs[: min(2, len(recovery_outputs))]
            ]
            recovered = self.refiner.refine(
                learned_candidates,
                frame,
                detection,
                mode,
                motion_reference,
                occluder_poses,
            )
            # Preserve the original learned-recovery ranking behavior. Pose
            # de-duplication is only needed when optional search stages are
            # merged below.
            ranked = sorted(
                [*ranked, *recovered],
                key=lambda item: item.score.total_error,
            )
        return ranked

    def _update_track_appearance(
        self,
        track: Track,
        frame: FrameData,
        detection: Detection,
    ) -> None:
        if not self.config.association.identity_enabled:
            return
        descriptor = detection_appearance_descriptor(frame.image, detection)
        if descriptor is None:
            return
        if (
            track.appearance_descriptor is None
            or track.appearance_descriptor.shape != descriptor.shape
        ):
            track.appearance_descriptor = descriptor.copy()
            return
        momentum = self.config.association.appearance_momentum
        updated = (
            momentum * track.appearance_descriptor
            + (1.0 - momentum) * descriptor
        )
        total = float(updated.sum())
        track.appearance_descriptor = (
            updated / total if total > 0 else descriptor.copy()
        )

    def _track_one(
        self,
        frame: FrameData,
        detection: Detection,
        track: Track | None,
        fresh: list[PoseHypothesis],
        occluder_poses: tuple[np.ndarray, ...] = (),
    ) -> TrackedInstanceResult | None:
        started = time.perf_counter()
        mode = self._mode_for(track)
        periodic_global = self._periodic_global(track)
        flow_measurement = self._flow_measurement(track, frame)
        motion_reference = (
            self.generator.propagated_pose(track) if track is not None else None
        )
        ranked = self._search(
            frame=frame,
            detection=detection,
            track=track,
            fresh=fresh,
            mode=mode,
            periodic_global=periodic_global,
            flow_measurement=flow_measurement,
            motion_reference=motion_reference,
            occluder_poses=occluder_poses,
        )
        if not ranked:
            return None

        recovery_modes: list[TrackMode] = []
        recovery_config = self.config.same_frame_recovery
        if recovery_config.enabled and track is not None:
            provisional_mode = self._next_mode(
                self._confidence_with_margin(ranked)
            )
            if (
                provisional_mode == TrackMode.UNCERTAIN
                and mode == TrackMode.NORMAL
                and recovery_config.retry_uncertain
            ):
                recovery_modes.append(TrackMode.UNCERTAIN)
            if (
                provisional_mode == TrackMode.LOST
                and mode != TrackMode.LOST
                and recovery_config.retry_lost
            ):
                if (
                    mode == TrackMode.NORMAL
                    and recovery_config.retry_uncertain
                ):
                    recovery_modes.append(TrackMode.UNCERTAIN)
                recovery_modes.append(TrackMode.LOST)

        recovery_used: list[TrackMode] = []
        for recovery_mode in recovery_modes:
            retry_ranked = self._search(
                frame=frame,
                detection=detection,
                track=track,
                fresh=(
                    fresh if recovery_config.force_global else []
                ),
                mode=recovery_mode,
                periodic_global=recovery_config.force_global,
                flow_measurement=flow_measurement,
                motion_reference=motion_reference,
                occluder_poses=occluder_poses,
            )
            if retry_ranked:
                ranked = self._merge_ranked(ranked, retry_ranked)
                recovery_used.append(recovery_mode)
            if self._next_mode(
                self._confidence_with_margin(ranked)
            ) != TrackMode.LOST:
                # An uncertain retry already recovered useful image support;
                # avoid paying for the lost-state search unless still lost.
                if recovery_mode == TrackMode.UNCERTAIN:
                    break

        chosen = ranked[0]
        confidence = self._confidence_with_margin(ranked)
        next_mode = self._next_mode(confidence)
        global_used = "global_rank_" in chosen.hypothesis.source or (
            "|learned_recovery" in chosen.hypothesis.source
            and "global_rank_" in chosen.hypothesis.source
        )
        if track is None:
            track = Track(
                track_id=self.next_track_id,
                obj_id=chosen.hypothesis.obj_id,
                pose=chosen.hypothesis.pose.copy(),
                previous_pose=None,
                bbox_xywh=detection.bbox_xywh.copy(),
                confidence=confidence,
                mode=next_mode,
                source=chosen.hypothesis.source,
                scene_id=frame.scene_id,
                im_id=frame.im_id,
                external_id=detection.external_id,
            )
            self.next_track_id += 1
        else:
            track.previous_pose = track.pose.copy()
            track.pose = chosen.hypothesis.pose.copy()
            track.bbox_xywh = detection.bbox_xywh.copy()
            track.confidence = confidence
            track.mode = next_mode
            track.source = chosen.hypothesis.source
            track.scene_id = frame.scene_id
            track.im_id = frame.im_id
            if detection.external_id is not None:
                track.external_id = detection.external_id
            track.age += 1
            track.missed = 0
        track.frames_since_global = 0 if global_used else track.frames_since_global + 1
        track.previous_gray = frame.gray.copy()
        track.previous_mask = detection.mask.copy()
        self._update_track_appearance(track, frame, detection)
        track.last_score = chosen.score
        track.history.append(track.pose.copy())
        if len(track.history) > 64:
            del track.history[:-64]
        self.tracks[track.track_id] = track
        return TrackedInstanceResult(
            track=track,
            detection=detection,
            chosen=chosen,
            alternatives=ranked[1:],
            periodic_global=periodic_global,
            elapsed_s=time.perf_counter() - started,
            same_frame_recovery=bool(recovery_used),
            same_frame_recovery_mode=(
                recovery_used[-1].value if recovery_used else ""
            ),
            occluders_used=len(occluder_poses),
        )

    def process_frame(
        self,
        frame: FrameData,
        fresh_groups: list[list[PoseHypothesis]],
    ) -> TrackingFrameResult:
        if (
            self.config.reset_on_scene_change
            and self.current_scene_id is not None
            and frame.scene_id != self.current_scene_id
        ):
            self.reset()
        self.current_scene_id = frame.scene_id
        active_tracks = list(self.tracks.values())
        matches, unmatched_track_indices, unmatched_detection_indices = associate_tracks(
            active_tracks,
            frame.detections,
            frame.image.shape[:2],
            self.config.association,
            frame.image,
        )
        fresh_by_detection = assign_prediction_groups(
            fresh_groups, frame.detections, frame.K, frame.image.shape[:2]
        )
        track_by_detection = {
            detection_index: active_tracks[track_index]
            for track_index, detection_index in matches
        }
        occluder_pose_by_detection: dict[int, np.ndarray] = {}
        if self.config.occlusion.enabled:
            for detection_index in range(len(frame.detections)):
                matched_track = track_by_detection.get(detection_index)
                if matched_track is not None:
                    occluder_pose_by_detection[detection_index] = (
                        self.generator.propagated_pose(matched_track)
                    )
                    continue
                fresh = fresh_by_detection.get(detection_index, [])
                if fresh:
                    occluder_pose_by_detection[detection_index] = (
                        fresh[0].pose.copy()
                    )

        def occluders_for(detection_index: int) -> tuple[np.ndarray, ...]:
            if not self.config.occlusion.enabled:
                return ()
            return tuple(
                pose
                for other_index, pose in occluder_pose_by_detection.items()
                if other_index != detection_index
            )

        result = TrackingFrameResult(frame.scene_id, frame.im_id)
        for track_index, detection_index in matches:
            tracked = self._track_one(
                frame,
                frame.detections[detection_index],
                active_tracks[track_index],
                fresh_by_detection.get(detection_index, []),
                occluders_for(detection_index),
            )
            if tracked is not None:
                result.instances.append(tracked)

        # New or currently unassociated detections require a global hypothesis.
        for detection_index in unmatched_detection_indices:
            fresh = fresh_by_detection.get(detection_index, [])
            tracked = self._track_one(
                frame,
                frame.detections[detection_index],
                None,
                fresh,
                occluders_for(detection_index),
            )
            if tracked is None:
                result.unmatched_detection_ids.append(
                    frame.detections[detection_index].detection_id
                )
            else:
                result.instances.append(tracked)

        for track_index in unmatched_track_indices:
            track = active_tracks[track_index]
            track.missed += 1
            track.mode = TrackMode.LOST
            if track.missed >= self.config.state.delete_after_misses:
                self.tracks.pop(track.track_id, None)
                result.deleted_track_ids.append(track.track_id)
        return result

    @staticmethod
    def diagnostics(result: TrackingFrameResult) -> list[dict[str, Any]]:
        rows = []
        for instance in result.instances:
            chosen = instance.chosen
            rows.append(
                {
                    "scene_id": result.scene_id,
                    "im_id": result.im_id,
                    "track_id": instance.track.track_id,
                    "detection_id": instance.detection.detection_id,
                    "mode": instance.track.mode.value,
                    "source": chosen.hypothesis.source,
                    "confidence": instance.track.confidence,
                    "total_error": chosen.score.total_error,
                    "silhouette_iou": chosen.score.silhouette_iou,
                    "bbox_iou": chosen.score.bbox_iou,
                    "edge_error": chosen.score.edge_error,
                    "depth_error": chosen.score.depth_error,
                    "motion_error": chosen.score.motion_error,
                    "alternatives": len(instance.alternatives),
                    "periodic_global": int(instance.periodic_global),
                    "same_frame_recovery": int(
                        instance.same_frame_recovery
                    ),
                    "same_frame_recovery_mode": (
                        instance.same_frame_recovery_mode
                    ),
                    "occluders_used": instance.occluders_used,
                    "elapsed_s": instance.elapsed_s,
                }
            )
        return rows
