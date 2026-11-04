"""Print interpreter and library versions for the two remote task environments."""
import subprocess
import sys

CODE = (
    'import sys; print(sys.version.split()[0]);\n'
    'import numpy; print("numpy", numpy.__version__);\n'
    'try:\n'
    '    import torch; print("torch", torch.__version__)\n'
    'except Exception as exc: print("torch missing", exc)\n'
    'try:\n'
    '    import jittor; print("jittor", jittor.__version__)\n'
    'except Exception as exc: print("jittor missing", exc)\n'
)

for label, python in (("source", "/root/autodl-tmp/envs/source/bin/python"),
                      ("target", "/root/autodl-tmp/envs/target/bin/python")):
    print("==", label, python)
    result = subprocess.run([python, "-c", CODE], text=True, capture_output=True)
    print(result.stdout.strip())
    if result.returncode:
        print("stderr:", result.stderr.strip()[-500:])
