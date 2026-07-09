"""Training dataset that exposes geometry needed by pose-aware losses.

The production AssettoCorsaFineTuneSet deliberately returns only the tensors
needed by the original GigaPose objectives. This subclass leaves all of that
processing intact and adds the source-template and target ground-truth poses.
"""

from __future__ import annotations

from fine_tuning.dataloader import AssettoCorsaFineTuneSet


class PoseAwareAssettoCorsaFineTuneSet(AssettoCorsaFineTuneSet):
    """Assetto Corsa loader with template and target poses in each batch."""

    def process_keypoints(
        self,
        real_data,
        template_data,
        T_real2template,
        T_template2real,
    ):
        # collate_fn is synchronous, so this geometry belongs to the batch
        # currently being assembled. Clone it because the parent processing
        # continues after this method returns.
        self._pose_aware_src_pose = template_data.pose.clone()
        self._pose_aware_tar_pose = real_data.pose.clone()
        return super().process_keypoints(
            real_data,
            template_data,
            T_real2template,
            T_template2real,
        )

    def collate_fn(self, batch):
        try:
            output = super().collate_fn(batch)
            if output is None:
                return None
            output.register_tensor("src_pose", self._pose_aware_src_pose)
            output.register_tensor("tar_pose", self._pose_aware_tar_pose)
            return output
        finally:
            if hasattr(self, "_pose_aware_src_pose"):
                del self._pose_aware_src_pose
            if hasattr(self, "_pose_aware_tar_pose"):
                del self._pose_aware_tar_pose
