import numpy as np
from pathlib import Path

rot_dir = Path("/path/to/result1_good_rotation/predictions")
trans_dir = Path("/path/to/result2_good_translation/predictions")
out_dir = Path("/home/anahita/gigapose/gigaPose_datasets/results/ensembled_rotation_translation/predictions")

out_dir.mkdir(parents=True, exist_ok=True)

for rot_file in sorted(rot_dir.glob("*.npz")):
    trans_file = trans_dir / rot_file.name

    if not trans_file.exists():
        print(f"Missing matching file: {trans_file}")
        continue

    rot_data = dict(np.load(rot_file, allow_pickle=True))
    trans_data = dict(np.load(trans_file, allow_pickle=True))

    T_rot = rot_data["poses"]
    T_trans = trans_data["poses"]

    if T_rot.shape != T_trans.shape:
        print(f"Shape mismatch: {rot_file.name}: {T_rot.shape} vs {T_trans.shape}")
        continue

    T_final = T_rot.copy()

    # rotation from result1
    T_final[:, :, :3, :3] = T_rot[:, :, :3, :3]

    # translation from result2
    T_final[:, :, :3, 3] = T_trans[:, :, :3, 3]

    rot_data["poses"] = T_final

    # optional: keep scores from rotation-good model
    # rot_data["scores"] = trans_data["scores"]  # use this instead if result2 scores are better

    np.savez_compressed(out_dir / rot_file.name, **rot_data)

print("Done.")
print("Saved merged predictions to:", out_dir)