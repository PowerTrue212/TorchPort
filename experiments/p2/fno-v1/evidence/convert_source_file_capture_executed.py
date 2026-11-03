"""Save the official converter's exact output even when it is invalid Python."""

import argparse
import hashlib
import json
from pathlib import Path

from jittor.utils.pytorch_converter import convert

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = args.source.read_bytes()
converted = convert(source.decode("utf-8"))
output = converted.encode("utf-8")
args.output.write_bytes(output)
result = {
    "source": str(args.source),
    "source_sha256": hashlib.sha256(source).hexdigest(),
    "output": str(args.output),
    "output_sha256": hashlib.sha256(output).hexdigest(),
    "output_bytes": len(output),
}
try:
    compile(converted, str(args.output), "exec")
except SyntaxError as exc:
    result.update({
        "compiles": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_line": exc.lineno,
        "error_text": exc.text,
    })
else:
    result["compiles"] = True
print(json.dumps(result, ensure_ascii=False))
if not result["compiles"]:
    raise SystemExit(1)
