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

- **`The first run on TensorRT EP takes about 15 minutes to compile ONNX to TensorRT Engine. Anyone who can't use this environment to its fullest should stay away.`**
- All processing and models are optimized for TensorRT, which is very slow on CPU and CUDA.
- Because of the N batches x M batches variable batch input model, CUDA is extremely slow due to the frequent GPU initialization process.
- Environment
  - onnx==1.15.0
  - onnxruntime-gpu==1.16.1 (TensorRT EP builtin)
  - numpy==1.24.3
  - scipy==1.10.1
  - opencv-contrib-python==4.9.0.80
  - pycuda==2022.2
  - onnx-tensorrt==release/8.5-GA
    - Tricks with docker build 
      ```dockerfile
      # Install onnx-tensorrt
      RUN git clone -b release/8.5-GA --recursive https://github.com/onnx/onnx-tensorrt ../onnx-tensorrt \
          && pushd ../onnx-tensorrt \
          && mkdir build \
          && pushd build \
          && cmake .. -DTENSORRT_ROOT=/usr/src/tensorrt \
          && make -j$(nproc) \
          && sudo make install \
          && popd \
          && popd \
          && pip install onnx==${ONNXVER} \
          && pip install pycuda==${PYCUDAVER} \
          && echo 'pushd ../onnx-tensorrt > /dev/null' >> ~/.bashrc \
          # At docker build time, setup.py fails because NVIDIA's physical GPU device cannot be detected.
          # Therefore, a workaround is applied to configure setup.py to run on first access.
          # By Katsuya Hyodo
          && echo 'python setup.py install --user 1>/dev/null 2>/dev/null' >> ~/.bashrc \
          && echo 'popd > /dev/null' >> ~/.bashrc \
          && echo 'export CUDA_MODULE_LOADING=LAZY' >> ~/.bashrc \
          && echo 'export PATH=${PATH}:/usr/src/tensorrt/bin:${HOME}/onnx-tensorrt/build' >> ~/.bashrc
      ```

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

- YOLOX INPUTs/OUTPUTs/Custom post-process

  |INPUTs/OUTPUTs/Post-Process|Note|
  |:-:|:-|
  |![20240103191712](https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/assets/33194443/fa58c24f-69ee-4e9e-99f3-f2b93306787e)|・INPUTs<br>`input`: Entire image [1,3,H,W]<br><br>・OUTPUTs<br>`batchno_classid_score_x1y1x2y2`: [N,[batchNo,classid,score,x1,y1,x2,y2]]. Final output with NMS implemented|

- ReID INPUTs/OUTPUTs

  |INPUTs/OUTPUTs|Note|
  |:-:|:-|
  |![20240103190410](https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/assets/33194443/3d05ff23-d379-4a6a-a762-50efa62a4ab4)|・INPUTs<br>`base_images`: N human images<br>`target_features`: M human features extracted in the previous inference<br><br>・OUTPUTs<br>`similarities`: COS similarity between N human features and target_features<br>`base_features`: N human features|

- ReID custom post-process

  |Custom Post-Process|Note|
  |:-:|:-|
  |![image](https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/assets/33194443/8bf44dec-9b00-4d9b-8aa6-e4f7087b3deb)|`A`: Normalized section<br>`B`: COS similarity calculation section|

## Acknowledgments

- https://github.com/NirAharon/BoT-SORT
  - https://arxiv.org/abs/2206.14651
- https://github.com/JDAI-CV/fast-reid
