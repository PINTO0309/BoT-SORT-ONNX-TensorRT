# BoT-SORT-ONNX-TensorRT
BoT-SORT + YOLOX implemented using only onnxruntime, Numpy and scipy, without cython_bbox and PyTorch. Fast human tracker.

```bash
docker pull pinto0309/botsort_onnx_tensorrt:latest

# With USBCam
xhost +local: && \
docker run --rm -it --gpus all \
-v `pwd`:/workdir \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
pinto0309/botsort_onnx_tensorrt:latest

# Without USBCam
xhost +local: && \
docker run --rm -it --gpus all \
-v `pwd`:/workdir \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
pinto0309/botsort_onnx_tensorrt:latest
```
```
# ONNX files are downloaded automatically.
usage: demo_bottrack_onnx_tfite.py \
  [-h] \
  [-odm {
    yolox_n_body_head_hand_post_0461_0.4428_1x3x384x640.onnx,
    yolox_t_body_head_hand_post_0299_0.4522_1x3x384x640.onnx,
    yolox_s_body_head_hand_post_0299_0.4983_1x3x384x640.onnx,
    yolox_m_body_head_hand_post_0299_0.5263_1x3x384x640.onnx,
    yolox_l_body_head_hand_post_0299_0.5420_1x3x384x640.onnx,
    yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx
  }] \
  [-fem {
    mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx,
    mot17_sbs_S50_NMx3x288x128_post_feature_only.onnx,
    mot17_sbs_S50_NMx3x320x128_post_feature_only.onnx,
    mot17_sbs_S50_NMx3x352x128_post_feature_only.onnx,
    mot17_sbs_S50_NMx3x384x128_post_feature_only.onnx,
    mot20_sbs_S50_NMx3x256x128_post_feature_only.onnx,
    mot20_sbs_S50_NMx3x288x128_post_feature_only.onnx,
    mot20_sbs_S50_NMx3x320x128_post_feature_only.onnx,
    mot20_sbs_S50_NMx3x352x128_post_feature_only.onnx,
    mot20_sbs_S50_NMx3x384x128_post_feature_only.onnx
  }] \
  [-tc TRACK_TARGET_CLASSES [TRACK_TARGET_CLASSES ...]] \
  [-v VIDEO] \
  [-ep {cpu,cuda,tensorrt}] \
  [-dvw]

options:
  -h, --help
    show this help message and exit
  -odm {...}, --object_detection_model {...}
    ONNX/TFLite file path for YOLOX.
  -fem {...}, --feature_extractor_model {...}
    ONNX/TFLite file path for FastReID.
  -tc TRACK_TARGET_CLASSES [TRACK_TARGET_CLASSES ...], \
    --track_target_classes TRACK_TARGET_CLASSES [TRACK_TARGET_CLASSES ...]
    List of class IDs to be tracked. 0:Body, 1: Head, 2: Hand
  -v VIDEO, --video VIDEO
    Video file path or camera index.
  -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
    Execution provider for ONNXRuntime.
  -dvw, --disable_video_writer
    Disable video writer. Eliminates the file I/O load associated with automatic recording
    to MP4. Devices that use a MicroSD card or similar for main storage can speed up overall
    processing.
```
```bash
# ONNX files are downloaded automatically.
python demo_bottrack_onnx_tflite.py -v 0

python demo_bottrack_onnx_tflite.py -v xxxx.mp4
```

**`The first run on TensorRT EP takes about 15 minutes to compile ONNX to TensorRT Engine. Anyone who can't use this environment to its fullest should stay away.`**
===

- onnxruntime + TensorRT 8.5.3 + YOLOX-X + BoT-SORT

  https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/assets/33194443/368bbfda-b204-4246-8663-259f999dab1c

- onnxruntime + TensorRT 8.5.3 + YOLOX-Nano + BoT-SORT

  https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/assets/33194443/647f6c0f-66c5-4213-b16c-fba534a0f2a6

- Models

   1. https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/tag/onnx
   2. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/426_YOLOX-Body-Head-Hand
   3. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/430_FastReID

- ONNX export + Validation (Fork from WWangYuHsiang/SMILEtrack)

  https://github.com/PINTO0309/SMILEtrack

- ONNX to TF/TFLite convert

  https://github.com/PINTO0309/onnx2tf

- Custom post-process

  |Custom Post-Process|Note|
  |:-:|:-|
  |![image](https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/assets/33194443/8bf44dec-9b00-4d9b-8aa6-e4f7087b3deb)|`A`: normalized section<br>`B`: COS similarity calculation section|

## Acknowledgments

- https://github.com/NirAharon/BoT-SORT
  - https://arxiv.org/abs/2206.14651
- https://github.com/JDAI-CV/fast-reid
