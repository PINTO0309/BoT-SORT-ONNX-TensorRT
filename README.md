# BoT-SORT-ONNX-TensorRT
BoT-SORT + YOLOX implemented using only onnxruntime, Numpy and scipy, without cython_bbox and PyTorch. Fast human tracker.

```bash
docker pull pinto0309/ubuntu22.04-cuda11.8-tensorrt8.5.3:latest

docker run --rm -it --gpus all \
-v `pwd`:/workdir \
pinto0309/ubuntu22.04-cuda11.8-tensorrt8.5.3
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

## Acknowledgments

- https://github.com/NirAharon/BoT-SORT
  - https://arxiv.org/abs/2206.14651
