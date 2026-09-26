# Real-rig camera calibration

`rs-enumerate-devices -c` dumps of the top camera (Intel RealSense D435) on each of the
evaluator's three rigs, captured 8 Sep 2026. The colour-stream intrinsics at 640x480, the mode
the evaluation captures, are the values `harness/isaac_trials.py` uses for the simulated top
camera (`RIG=rig1|rig2|rig6`). The runner has them as constants; these files are the source.
