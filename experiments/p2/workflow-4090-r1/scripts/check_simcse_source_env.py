"""Check the SimCSE source environment (transformers, torch) before building the task."""
import subprocess

CODE = (
    "import torch, transformers, numpy\n"
    "print('torch', torch.__version__)\n"
    "print('transformers', transformers.__version__)\n"
    "print('numpy', numpy.__version__)\n"
    "from transformers import BertConfig, BertModel\n"
    "config = BertConfig.from_json_file('/root/autodl-tmp/SimCSE-Benchmark-v1/assets/unsup-config.json')\n"
    "model = BertModel(config)\n"
    "names = list(model.state_dict())\n"
    "print('params', len(names))\n"
    "print('pooler', [n for n in names if n.startswith('pooler')])\n"
    "print('buffers', [n for n in names if 'position_ids' in n or 'mask' in n])\n"
)
result = subprocess.run(["/root/autodl-tmp/envs/source/bin/python", "-c", CODE],
                        text=True, capture_output=True)
print(result.stdout)
print(result.stderr[-2000:])
