"""Read-only Jittor API inventory for the controller LightGCN reference."""

import importlib.util

import jittor as jt

print({
    "jittor": jt.__version__,
    "sparse_names": [name for name in dir(jt) if "sparse" in name.lower()],
    "sparse_type": str(type(jt.sparse)),
    "sparse_members": [name for name in dir(jt.sparse) if not name.startswith("_")],
    "embedding_names": [name for name in dir(jt.nn) if "embedding" in name.lower()],
    "scipy_available": importlib.util.find_spec("scipy") is not None,
})
