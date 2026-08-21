from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from torchport.migrate import RepoCheck, RepositoryMigrator
from torchport.models import MigrationTarget

SIMPLE_MODEL = '''import torch
from torch import nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3)

    def forward(self, value):
        value = F.relu(self.conv(value))
        return torch.cat((value, value), dim=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyNet().to(device)
    value = torch.zeros((1, 3, 8, 8)).cuda()
    print(model(value).shape)


if __name__ == "__main__":
    main()
'''

MNIST_MODEL = '''import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


data = datasets.MNIST("data", download=True, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=8)
optimizer = optim.Adadelta([], lr=1.0)
scheduler = StepLR(optimizer, step_size=1)
'''

INIT_MODEL = '''import torch.nn as nn
import torch.nn.init as init


class InitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        init.normal_(self.fc.bias, std=.02)
        init.kaiming_normal_(self.fc.weight)
        init.orthogonal_(self.fc.weight, init.calculate_gain("relu"))

    def forward(self, value):
        return self.fc(value)
'''


class MigrationTests(unittest.TestCase):
    def _make_repository(self, root: Path) -> Path:
        repository = root / "source"
        repository.mkdir()
        (repository / "main.py").write_text(SIMPLE_MODEL, encoding="utf-8")
        (repository / "requirements.txt").write_text(
            "torch==2.6.0\nnumpy>=1.24\nPillow\n",
            encoding="utf-8",
        )
        return repository

    def test_native(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = self._make_repository(root)
            (repository / "model.pth").write_bytes(b"fixture")
            output = root / "native-output"

            report = RepositoryMigrator().migrate(
                repository,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
            )

            migrated = (output / "main.py").read_text(encoding="utf-8")
            ast.parse(migrated)
            self.assertEqual(report["status"], "converted")
            self.assertIn("import jittor as jt", migrated)
            self.assertIn("import jittor.nn as F", migrated)
            self.assertIn("def execute(self, value):", migrated)
            self.assertIn("nn.Conv(3, 4", migrated)
            self.assertIn("jt.concat", migrated)
            self.assertIn("bool(jt.flags.use_cuda)", migrated)
            self.assertNotIn(".to(device)", migrated)
            self.assertNotIn(".cuda()", migrated)
            requirements = (output / ".torchport" / "requirements.runtime.txt").read_text()
            self.assertIn("jittor>=1.3.11", requirements)
            self.assertIn("numpy<2", requirements)
            self.assertIn("Pillow", requirements)
            self.assertNotIn("torch==", requirements)
            manifest = json.loads(
                (output / ".torchport" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["summary"]["blockers"], 0)
            self.assertEqual(manifest["probe"]["model"], "TinyNet")
            self.assertEqual(manifest["probe"]["shape"], [1, 3, 8, 8])
            self.assertEqual(manifest["probe"]["weights"], "model.pth")
            self.assertTrue((output / ".torchport" / "probe.json").is_file())
            check = RepoCheck().check(output, runtime=False)
            self.assertTrue(check["passed"])

    def test_jtorch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = self._make_repository(root)
            output = root / "jtorch-output"

            report = RepositoryMigrator().migrate(
                repository,
                output,
                target=MigrationTarget.JTORCH,
            )

            self.assertEqual(report["status"], "converted")
            self.assertEqual((output / "main.py").read_text(), SIMPLE_MODEL)
            installer = (output / ".torchport" / "install.sh").read_text()
            self.assertIn("https://pypi.jittor.org/simple", installer)
            self.assertIn("--force-reinstall torch", installer)

    def test_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "source"
            repository.mkdir()
            (repository / "train.py").write_text(
                "import torch\nfrom torch.utils.data import RandomSampler\n",
                encoding="utf-8",
            )
            (repository / "kernel.cu").write_text("// custom op", encoding="utf-8")
            output = root / "output"

            report = RepositoryMigrator().migrate(
                repository,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
            )

            identifiers = {item["id"] for item in report["findings"]}
            self.assertEqual(report["status"], "partial")
            self.assertIn("repository.custom-operator", identifiers)
            self.assertIn("native.dataset-contract", identifiers)
            self.assertTrue((output / ".torchport" / "manifest.json").is_file())
            check = RepoCheck().check(output, runtime=False)
            self.assertFalse(check["passed"])

    def test_output_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = self._make_repository(root)
            migrator = RepositoryMigrator()

            with self.assertRaisesRegex(ValueError, "源仓库内部"):
                migrator.migrate(repository, repository / "converted")

            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "已经存在"):
                migrator.migrate(repository, existing)

    def test_portlib(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(MNIST_MODEL, encoding="utf-8")
            (source / "requirements.txt").write_text(
                "torch\ntorchvision\nPillow\n",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "partial")
            self.assertIn("probe.missing", {item["id"] for item in report["findings"]})
            self.assertIn("from portlib import datasets, transforms", migrated)
            self.assertIn("from portlib import StepLR", migrated)
            self.assertIn("from portlib import Adadelta", migrated)
            self.assertIn("portlib.DataLoader", migrated)
            self.assertTrue((root / "output" / "portlib.py").is_file())

    def test_init_portlib(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(INIT_MODEL, encoding="utf-8")

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("from portlib import init", migrated)
            self.assertNotIn("torch.nn.init", migrated)
            portlib = (root / "output" / "portlib.py").read_text(encoding="utf-8")
            self.assertIn("constant_", portlib)
            self.assertIn("kaiming_normal_", portlib)
            self.assertIn("normal_", portlib)
            self.assertIn("xavier_uniform_", portlib)
            self.assertIn("compat.nn-init", report["summary"]["automatic_rules"])
            self.assertTrue((root / "output" / "portlib.py").is_file())
            check = RepoCheck().check(root / "output", runtime=False)
            self.assertTrue(check["residue"]["passed"])

    def test_timm_vit_portlib(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from functools import partial
from torch import nn
from timm.models.vision_transformer import PatchEmbed, Block

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.patch = PatchEmbed(img_size=8, patch_size=4, embed_dim=16)
        self.block = Block(16, num_heads=2, norm_layer=norm)
        self.norm = norm(16)

    def forward(self, value):
        return self.norm(self.block(self.patch(value)))
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            portlib = (root / "output" / "portlib.py").read_text(encoding="utf-8")
            self.assertIn("import portlib", migrated)
            self.assertIn("from portlib import PatchEmbed, Block", migrated)
            self.assertIn("partial(portlib.LayerNorm", migrated)
            self.assertIn("class LayerNorm", portlib)
            self.assertIn("class PatchEmbed", portlib)
            self.assertIn("class Block", portlib)
            self.assertIn("self.drop = nn.Dropout", portlib)
            self.assertNotIn("self.drop1", portlib)
            self.assertIn("compat.layer-norm", report["summary"]["automatic_rules"])
            self.assertIn("compat.timm-vit", report["summary"]["automatic_rules"])

    def test_timm_layers_and_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        size = to_2tuple(2)
        self.drop = DropPath(0.0)
        self.weight = nn.Parameter(torch.zeros(1, size[0]))
        trunc_normal_(self.weight, std=.02)
        self.register_buffer("mask", torch.zeros(1, 2))

    @torch.jit.ignore
    def ignored(self):
        return {"mask"}

    def forward(self, value):
        return self.drop(value + self.mask)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 2], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            portlib = (root / "output" / "portlib.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("from portlib import DropPath, to_2tuple, trunc_normal_", migrated)
            self.assertIn("setattr(self, \"mask\", jt.zeros(1, 2)", migrated)
            self.assertIn("@portlib.identity", migrated)
            self.assertIn("def trunc_normal_", portlib)
            self.assertIn("def to_2tuple", portlib)
            self.assertIn("compat.jit-ignore", report["summary"]["automatic_rules"])
            self.assertIn("compat.timm-layers", report["summary"]["automatic_rules"])
            self.assertIn("module.register-buffer", report["summary"]["automatic_rules"])

    def test_checkpoint_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import torch.utils.checkpoint as checkpoint
from torch import nn

class Net(nn.Module):
    def forward(self, value):
        return checkpoint.checkpoint(lambda x: x + 1, value)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 2], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            portlib = (root / "output" / "portlib.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("import portlib as checkpoint", migrated)
            self.assertIn("def checkpoint", portlib)
            self.assertIn("compat.checkpoint", report["summary"]["automatic_rules"])

    def test_adaptive_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, value):
        return self.pool(value)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 2, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            portlib = (root / "output" / "portlib.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("portlib.AdaptiveAvgPool1d", migrated)
            self.assertIn("class AdaptiveAvgPool1d", portlib)
            self.assertIn("compat.adaptive-pool", report["summary"]["automatic_rules"])

    def test_bare_module_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch.nn import Module

class Net(Module):
    def forward(self, value):
        return value + 1
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 2], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("from jittor.nn import Module", migrated)
            self.assertIn("def execute", migrated)
            self.assertIn("module.forward-to-execute", report["summary"]["automatic_rules"])

    def test_einops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2)

    def forward(self, value):
        value = self.patch(value)
        value = rearrange(value, 'b n (h d) -> b h n d', h=2)
        return repeat(value[:, 0, 0], '... d -> b ... d', b=1)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 3, 4, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            portlib = (root / "output" / "portlib.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("from portlib import rearrange, repeat", migrated)
            self.assertIn("from portlib import Rearrange", migrated)
            self.assertIn("def rearrange", portlib)
            self.assertIn("class Rearrange", portlib)
            self.assertIn("compat.einops", report["summary"]["automatic_rules"])

    def test_model_scope_prunes_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import torch
from torch import nn

args = object()
torch.manual_seed(args.seed)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, value):
        return self.fc(value)

model = Net()

def train(loss):
    loss.backward()
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 2], "dtype": "float32"}],
                scope="model",
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertNotIn("args.seed", migrated)
            self.assertNotIn("def train", migrated)
            self.assertNotIn("loss.backward", migrated)
            self.assertIn("scope.prune-code", report["summary"]["automatic_rules"])

    def test_relu_inplace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, value):
        return self.act(value)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 1], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("nn.ReLU()", migrated)
            self.assertIn("api.nn.relu-inplace", report["summary"]["automatic_rules"])

    def test_autograd_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn
from torch.autograd import Variable

class Net(nn.Module):
    def forward(self, value):
        return Variable(value)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 1], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertNotIn("torch.autograd", migrated)
            self.assertNotIn("Variable(", migrated)
            self.assertIn("return value", migrated)
            self.assertIn("compat.autograd-variable", report["summary"]["automatic_rules"])

    def test_global_avg_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def forward(self, value):
        return F.avg_pool2d(value, value.size()[3])
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 1, 4, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("import portlib", migrated)
            self.assertIn("portlib.global_avg_pool2d(value)", migrated)
            self.assertIn("compat.global-avg-pool", report["summary"]["automatic_rules"])

    def test_parameter_stop_grad(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 4), requires_grad=False)

    def forward(self, value):
        return value + self.pos
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("jt.zeros(1, 4).stop_grad()", migrated)
            self.assertIn("parameters.stop-grad", report["summary"]["automatic_rules"])

    def test_numpy_float(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import numpy as np
from torch import nn

class Net(nn.Module):
    def forward(self, value):
        return value

dtype = np.float
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("dtype = float", migrated)
            self.assertIn("compat.numpy-float", report["summary"]["automatic_rules"])

    def test_data_copy_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = torch.zeros(1, 4)
        self.pos.data.copy_(torch.from_numpy(source()).float())

    def forward(self, value):
        return value + self.pos
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("self.pos.update(jt.array(source()).float())", migrated)
            self.assertIn("tensor.data-copy-update", report["summary"]["automatic_rules"])

    def test_view_to_reshape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn

class Net(nn.Module):
    def forward(self, value):
        return value.view([value.shape[0], -1])
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 2, 3], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("value.reshape([value.shape[0], -1])", migrated)
            self.assertIn("tensor.view-to-reshape", report["summary"]["automatic_rules"])

    def test_param_data_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.flat = self.fc.weight.data.view([self.fc.weight.shape[0], -1])

    def forward(self, value):
        return self.fc(value)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("self.fc.weight.reshape([self.fc.weight.shape[0], -1])", migrated)
            self.assertIn("tensor.param-data-alias", report["summary"]["automatic_rules"])

    def test_factory_device_keyword(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import torch
from torch import nn

class Net(nn.Module):
    def forward(self, value):
        return value + torch.rand(value.shape[0], value.shape[1], device=value.device)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("jt.rand(value.shape[0], value.shape[1])", migrated)
            self.assertNotIn("device=value.device", migrated)
            self.assertIn("device.factory-keyword", report["summary"]["automatic_rules"])

    def test_argsort_portlib(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import torch
from torch import nn

class Net(nn.Module):
    def forward(self, value):
        first = torch.argsort(value, dim=1)
        return torch.argsort(first, dim=1)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 4], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            portlib = (root / "output" / "portlib.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("import portlib", migrated)
            self.assertIn("portlib.argsort(value, dim=1)", migrated)
            self.assertIn("def argsort", portlib)
            self.assertIn("compat.argsort", report["summary"]["automatic_rules"])

    def test_reshape_shape_keyword(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn

class Net(nn.Module):
    def forward(self, value):
        return value.reshape(shape=(value.shape[0], -1))
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                inputs=[{"shape": [1, 2, 3], "dtype": "float32"}],
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            self.assertEqual(report["status"], "converted")
            self.assertIn("value.reshape((value.shape[0], -1))", migrated)
            self.assertIn("tensor.reshape-shape-keyword", report["summary"]["automatic_rules"])

    def test_probe_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn

class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, 2)

    def forward(self, value):
        return self.fc(value)

config = runtime_config()
model = Net(config.width)
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
            )

            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["probe"]["status"], "needs-args")
            blockers = [item for item in report["findings"] if item["severity"] == "blocker"]
            self.assertEqual(blockers[0]["id"], "probe.needs-args")

    def test_env(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "env"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "output"

            report = RepositoryMigrator().migrate(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                env={"TP_WIDTH": "4"},
            )

            self.assertEqual(report["status"], "converted")
            self.assertEqual(
                report["probe"]["env"],
                {"TP_WIDTH": "4", "TP_HIDDEN": None},
            )
            runner = (output / ".torchport" / "run.py").read_text(encoding="utf-8")
            self.assertIn("'TP_WIDTH': '4'", runner)
            self.assertIn("'TP_HIDDEN': None", runner)

    def test_probe_override(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "dynamic"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "output"

            report = RepositoryMigrator().migrate(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="DynNet",
                inputs=[{"shape": [1, 4], "dtype": "float32"}],
                model_args=[4, 6],
            )

            self.assertEqual(report["status"], "converted")
            self.assertEqual(report["probe"]["status"], "ready")
            self.assertEqual(report["probe"]["args"], [4, 6])
            self.assertEqual(report["probe"]["shape"], [1, 4])
            identifiers = {item["id"] for item in report["findings"]}
            self.assertNotIn("probe.needs-args", identifiers)
            saved = json.loads(
                (output / ".torchport" / "probe.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["model"], "DynNet")
            self.assertEqual(saved["inputs"][0]["shape"], [1, 4])

    def test_explicit_entrypoint_keeps_probe_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            requested = source / "target"
            other = source / "other"
            requested.mkdir(parents=True)
            other.mkdir()
            (requested / "model.py").write_text(
                """from torch import nn

class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, value):
        return self.fc(value)
""",
                encoding="utf-8",
            )
            (other / "model.py").write_text(
                """from torch import nn

class Net(nn.Module):
    def forward(self, value):
        return value
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="target/model.py",
                model="Net",
                model_args=[4],
                inputs=[{"shape": [1, 4], "dtype": "float32"}],
                scope="model",
            )

            self.assertEqual(report["status"], "converted")
            self.assertEqual(report["entrypoint"], "target/model.py")
            self.assertEqual(report["probe"]["entry"], "target/model.py")
            self.assertEqual(report["scope"]["files"], ["target/model.py"])

    def test_model_scope_downgrades_outside_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "model.py").write_text(SIMPLE_MODEL, encoding="utf-8")
            (source / "data.py").write_text(
                "from torch.utils.data import RandomSampler\n",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="model.py",
                scope="model",
            )

            self.assertEqual(report["status"], "converted")
            self.assertEqual(report["scope"]["files"], ["model.py"])
            data_findings = [item for item in report["findings"] if item["file"] == "data.py"]
            self.assertEqual(data_findings[0]["severity"], "warning")
            self.assertEqual(data_findings[0]["scoped"], "outside-model")
            check = RepoCheck().check(root / "output", runtime=False)
            self.assertTrue(check["passed"])
            self.assertEqual(check["syntax"]["files"], 2)

    def test_model_scope_prunes_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet

class Net(nn.Module):
    def forward(self, value):
        return value

def make():
    return Net()

def other():
    model = ResNet()
    return load_state_dict_from_url("x"), model
""",
                encoding="utf-8",
            )

            report = RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="make",
                shapes=((1, 1),),
                scope="model",
            )

            migrated = (root / "output" / "main.py").read_text(encoding="utf-8")
            blockers = [item for item in report["findings"] if item["severity"] == "blocker"]
            scoped = {item.get("scoped") for item in report["findings"]}
            self.assertEqual(report["status"], "converted")
            self.assertEqual(blockers, [])
            self.assertNotIn("torch.hub", migrated)
            self.assertNotIn("torchvision.models", migrated)
            self.assertIn("scope.prune-import", report["summary"]["automatic_rules"])
            self.assertIn("outside-symbol", scoped)
            check = RepoCheck().check(root / "output", runtime=False)
            self.assertTrue(check["residue"]["passed"])

    def test_check_imports_package_entry_as_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            package = source / "pkg"
            torch_pkg = source / "torch"
            package.mkdir(parents=True)
            torch_pkg.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "parts.py").write_text("VALUE = 1\n", encoding="utf-8")
            (torch_pkg / "__init__.py").write_text(
                """class Module:
    pass

class NN:
    Module = Module

nn = NN()
""",
                encoding="utf-8",
            )
            (package / "model.py").write_text(
                """from .parts import VALUE
from torch import nn

class Net(nn.Module):
    def forward(self, value):
        return value
""",
                encoding="utf-8",
            )

            RepositoryMigrator().migrate(
                source,
                root / "output",
                target=MigrationTarget.JTORCH,
                entrypoint="pkg/model.py",
                model="Net",
                inputs=[{"shape": [1, 1], "dtype": "float32"}],
            )
            check = RepoCheck().check(root / "output", runtime=True)

            self.assertTrue(check["passed"])
            self.assertEqual(check["runtime"]["mode"], "import")


if __name__ == "__main__":
    unittest.main()
