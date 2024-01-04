import subprocess
import re

def get_nvidia_gpu_model():
    try:
        # nvidia-smiコマンドを実行
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)

        # GPUの型番を正規表現で抽出
        models = re.findall(r'GPU \d+: (.*?)(?= \(UUID)', output)
        return models
    except Exception as e:
        print(f"Error: {e}")
        return []

# GPUの型番を取得して表示
gpu_models = get_nvidia_gpu_model()
print("NVIDIA GPU Models:", gpu_models)