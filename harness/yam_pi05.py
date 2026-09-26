"""Reconstructed `yam_pi05` openpi config for robocurve/pi05-yam-molmoact2.

The published checkpoint documents its interface but the TrainConfig that openpi
needs to load it lives in github.com/robocurve/pi05-yam-replication, which the model
card itself marks private (and which 404s). Everything below is rebuilt from what the
card DOES state:

  "images": {"top", "left", "right"}   HWC uint8, 360x640 source
  "state" : (14,) absolute joint positions, 2x (6 joints + gripper)
  "actions": (16, 14) absolute joint positions at 30 fps
  normalization: quantile (q01/q99), vendored at assets/yam-bimanual-merged/norm_stats.json
  base: openpi pi05, pinned at 15a9616a00943ada6c20a0f158e3adb39df2ccac

YAM is 14-dim bimanual with three cameras, exactly like Aloha, so the internal image
keys map straight across. The one thing that must NOT be reused from Aloha is
`adapt_to_pi` -- that applies Aloha-specific joint/gripper space conversion
(_joint_flip_mask etc.) which would corrupt YAM joints.
"""
import dataclasses, numpy as np
from typing import ClassVar
from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top", "left", "right")

    def __call__(self, data: dict) -> dict:
        imgs = data["images"]
        extra = set(imgs) - set(self.EXPECTED_CAMERAS)
        if extra:
            raise ValueError(f"unexpected cameras {extra}; expected {self.EXPECTED_CAMERAS}")
        base = imgs["top"]
        images = {"base_0_rgb": base}
        masks = {"base_0_rgb": np.True_}
        for dest, src in (("left_wrist_0_rgb", "left"), ("right_wrist_0_rgb", "right")):
            if src in imgs:
                images[dest] = imgs[src]; masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base); masks[dest] = np.False_
        out = {"image": images, "image_mask": masks, "state": np.asarray(data["state"])}
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}


def yam_pi05_config(assets_dir: str) -> _config.TrainConfig:
    return _config.TrainConfig(
        name="yam_pi05",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=16),
        data=_config.SimpleDataConfig(
            repo_id="allenai/MolmoAct2-BimanualYAM-Dataset",
            assets=_config.AssetsConfig(assets_dir=assets_dir, asset_id="yam-bimanual-merged"),
            base_config=_config.DataConfig(prompt_from_task=True, use_quantile_norm=True),
            data_transforms=lambda model: transforms.Group(
                inputs=[YamInputs()], outputs=[YamOutputs()]),
        ),
    )
