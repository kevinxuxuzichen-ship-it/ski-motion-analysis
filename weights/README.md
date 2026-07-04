# Model Weights

## Files

| File | Description |
|------|-------------|
| `best.pt` | PyTorch pose model (YOLO 24-keypoint, incl. skis & poles) |
| `best_float32.tflite` | TFLite export for on-device (mobile) inference |
| `best_float16.tflite` | Half-precision TFLite export |

## Model Details

- **Task**: 2D pose estimation, 24 keypoints (body joints + skis + poles)
- **Base**: YOLO-pose (nano)
- **Validation**: Pose mAP@50 ≈ 0.96 on the Ski2DPose validation split

## Reproduction

To retrain from scratch, obtain the Ski2DPose dataset directly from EPFL and follow the
training notebook in `../notebooks/`.
