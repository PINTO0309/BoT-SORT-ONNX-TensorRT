# BoT-SORT-ONNX-TensorRT
BoT-SORT + YOLOX implemented using only onnxruntime, Numpy and scipy, without cython_bbox and PyTorch. Fast human tracker.

This repository does not use the less accurate CNN Feature Extractor. Instead, use Transformer's Fast-ReID Feature Extractor.

https://github.com/PINTO0309/PINTO_model_zoo/tree/main/430_FastReID#4-similarity-validation

Tolerance with respect to occlusion would be considerably more accurate using ByteTrack's MOT17 model than using my YOLOX model. However, I did not include ByteTrack's object detection model in this repository because I wanted to detect body parts other than the whole body at the same time with high accuracy.

https://github.com/ifzhang/ByteTrack

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
```bash
# ONNX files are downloaded automatically.
python demo_bottrack_onnx_tflite.py -v 0

python demo_bottrack_onnx_tflite.py -v xxxx.mp4
```

https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/assets/33194443/097ee727-2279-4eb1-9958-49ae4417c75c

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

- **`The first run on TensorRT EP takes about 15 minutes to compile ONNX to TensorRT Engine. Anyone who can't use this environment to its fullest should stay away.`**
- All processing and models are optimized for TensorRT, which is very slow on CPU and CUDA.
- Because of the N batches x M batches variable batch input model, CUDA is extremely slow due to the frequent GPU initialization process.
- The pre-optimized TensorRT Engine for RTX 30xx (Compute Capability 8.6) is automatically downloaded. If you are using a GPU model number other than RTX 30xx (Compute Capability 8.6), you will need to optimize the TensorRT Engine for each GPU model. https://developer.nvidia.com/cuda-gpus
- TensorRT Engine optimization
  ```bash
  pip install -U sit4onnx
  # It takes about 221 seconds.
  ./optimize_od_tensorrt_engine.sh yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx
  # It takes about 24,284 seconds. (More than 6 hours and 30 minutes.)
  ./optimize_reid_tensorrt_engine.sh mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx
  ```
- Environment
  - onnx==1.15.0
  - onnxruntime-gpu==1.16.1 (TensorRT EP builtin)
  - sit4onnx==1.0.7
  - numpy==1.24.3
  - scipy==1.10.1
  - opencv-contrib-python==4.9.0.80
  - requests==2.31.0
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

- onnxruntime + TensorRT 8.5.3 + YOLOX-X + BoT-SORT + USBCamera

  https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/assets/33194443/5523ae9f-cae5-4734-83c9-9931f01dd2c8

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

- Adjustment of YOLOX NMS parameters

  Because I add my own post-processing to the end of the model, which can be inferred by TensorRT, CUDA, and CPU, the benchmarked inference speed is the end-to-end processing speed including all pre-processing and post-processing. EfficientNMS in TensorRT is very slow and should be offloaded to the CPU. Also, the ONNX and TensorRT Engine published in this repository are optimized for my hobby use, so the score threshold, IoU threshold, and maximum number of output boxes need to be adjusted according to your requirements. If the detection performance of YOLOX-X is very high, but the number of bounding box outputs for object detection seems low, you may need to adjust the maximum number of output boxes or score threshold.

  - NMS default parameter

    |param|value|note|
    |:-|-:|:-|
    |max_output_boxes_per_class|20|Maximum number of outputs per class of one type. `20` indicates that the maximum number of people detected is `20`, the maximum number of heads detected is `20`, and the maximum number of hands detected is `20`. The larger the number, the more people can be detected, but the inference speed slows down slightly due to the larger overhead of NMS processing by the CPU. In addition, as the number of elements in the final output tensor increases, the amount of information transferred between hardware increases, resulting in higher transfer costs on the hardware circuit. Therefore, it would be desirable to set the numerical size to the minimum necessary.|
    |iou_threshold|0.40|A value indicating the percentage of occlusion allowed for multiple bounding boxes of the same class. `0.40` is excluded from the detection results if, for example, two bounding boxes overlap in more than 41% of the area. The smaller the value, the more occlusion is tolerated, but over-detection may increase.|
    |score_threshold|0.25|Bounding box confidence threshold. Specify in the range of `0.00` to `1.00`. The larger the value, the stricter the filtering and the lower the NMS processing load, but in exchange, all but bounding boxes with high confidence values are excluded from detection. This is a parameter that has a very large percentage impact on NMS overhead.|

  - Change NMS parameters

    Use **[PINTO0309/sam4onnx](https://github.com/PINTO0309/sam4onnx)** to rewrite the `NonMaxSuppression` parameter in the ONNX file.

    For example,
    ```bash
    pip install onnxsim==0.4.33 \
    && pip install -U simple-onnx-processing-tools \
    && pip install -U onnx \
    && python -m pip install -U onnx_graphsurgeon \
        --index-url https://pypi.ngc.nvidia.com

    ### max_output_boxes_per_class
    ### Example of changing the maximum number of detections per class to 100.
    sam4onnx \
    --op_name main01_nonmaxsuppression11 \
    --input_onnx_file_path yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx \
    --output_onnx_file_path yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression11 \
    --input_onnx_file_path yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx \
    --output_onnx_file_path yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression11 \
    --input_onnx_file_path yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx \
    --output_onnx_file_path yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```

## Acknowledgments

- https://github.com/NirAharon/BoT-SORT
  - https://arxiv.org/abs/2206.14651
- https://github.com/JDAI-CV/fast-reid
- https://github.com/ifzhang/ByteTrack
