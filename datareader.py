# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import json,os,sys

##################################
# TO load Our DATASET
##################################

from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import cv2

def load_rgb_middle_numpy(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Your images contain 3 side-by-side camera images.
    # Crop the middle one.
    third = w // 3
    img = img.crop((third, 0, 2 * third, h))

    return np.array(img, dtype=np.uint8)
  

def load_depth_numpy(depth_path):
    depth = Image.open(depth_path)
    depth_np = np.array(depth).astype(np.float32)

    # Your old loader used /256.0
    # depth_np = depth_np / 256.0
    depth_np = depth_np / 1000.0
    

    # If depth image is also triple-wide, crop middle.
    h, w = depth_np.shape[:2]
    if w / h > 4.0:
        third = w // 3
        depth_np = depth_np[:, third:2 * third]

    return depth_np
  
def load_intrinsics_numpy(path):
    return np.load(path).astype(np.float32)
  
  
def adjust_K_for_middle_crop(K, full_w):
    K = K.copy()
    third = full_w // 3
    K[0, 2] -= third   # shift cx left by one third
    return K
  

class RosbagFoundationPoseReader:
    def __init__(
        self,
        root,
        split_file,
        mask_file=None,
        use_lidar_mask=True,
        start_id=5020,
    ):
        self.root = Path(root)
        self.split_file = Path(split_file)
        self.mask_file = mask_file
        self.use_lidar_mask = use_lidar_mask
        self.start_id = start_id
        

        with open(self.split_file) as f:
            all_samples = [line.strip().split() for line in f if line.strip()]

        self.samples = []
        for sample in all_samples:
            img_rel = sample[0]
            stem = Path(img_rel).stem  # example: "005020"

            try:
                idx = int(stem)
            except ValueError:
                # If filename is like frame_005020.png
                idx = int("".join([c for c in stem if c.isdigit()]))

            if idx >= self.start_id:
                self.samples.append(sample)
        
        self.color_files = [str(self.root / s[0]) for s in self.samples]
        self.depth_files = [str(self.root / s[1]) for s in self.samples]
        self.K_files = [str(self.root / s[2]) for s in self.samples]

        self.id_strs = [
            Path(self.color_files[i]).stem for i in range(len(self.samples))
        ]

        # FoundationPose expects one K.
        # If your K changes per frame, we update it inside get_color/get_depth loop.
        img = Image.open(self.color_files[0]).convert("RGB")
        full_w, h = img.size
        third = full_w // 3
        self.K = load_intrinsics_numpy(self.K_files[0])
        self.K[0, 2] -= third

    def __len__(self):
        return len(self.samples)

    def get_color(self, i):
      img = Image.open(self.color_files[i]).convert("RGB")
      full_w, _ = img.size

      self.K = load_intrinsics_numpy(self.K_files[i])
      self.K = adjust_K_for_middle_crop(self.K, full_w)

      return load_rgb_middle_numpy(self.color_files[i])
    
    

    def get_depth(self, i):
        return load_depth_numpy(self.depth_files[i])

    def get_mask(self, i):
        """
        FoundationPose only needs this for the first frame.
        Option 1: load a manually-created mask.
        Option 2: create rough mask from valid LiDAR/depth pixels.
        """


        if self.mask_file is not None:
            mask = imageio.imread(self.mask_file)

            if mask.ndim == 3:
                mask = mask[..., 0]

            h, w = mask.shape[:2]
            if w / h > 4.0:
                third = w // 3
                mask = mask[:, third:2 * third]
                
            # resize to color/depth size
            target_h, target_w = self.get_depth(i).shape[:2]
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

            return mask > 0

        if self.use_lidar_mask:
            depth = self.get_depth(i)

            # valid sparse lidar pixels
            mask = depth > 0.001

            # Optional: dilate sparse lidar mask so FoundationPose gets a larger region
            mask_uint8 = mask.astype(np.uint8) * 255
            kernel = np.ones((15, 15), np.uint8)
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=2)

            return mask_uint8 > 0

        raise ValueError("Need either mask_file or use_lidar_mask=True")