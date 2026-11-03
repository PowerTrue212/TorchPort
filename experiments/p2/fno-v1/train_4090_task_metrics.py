"""Train the frozen FNO on bundled Darcy data and export test predictions."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(root / 'source'))
from torchport_train_adapter import build_optimizer, load_train_model  # noqa: E402

torch.manual_seed(20261030)
torch.cuda.manual_seed_all(20261030)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
model = load_train_model().cuda()
with np.load(root / 'train-inputs-4090-r2/initial-weights.npz') as archive:
    model.load_state_dict({name: torch.from_numpy(archive[name]).cuda() for name in archive.files})
optimizer = build_optimizer(model)
train_path = root / 'source/neuralop/data/datasets/data/darcy_train_16.pt'
test_path = root / 'source/neuralop/data/datasets/data/darcy_test_16.pt'
train = torch.load(train_path, map_location='cpu', weights_only=False)
test = torch.load(test_path, map_location='cpu', weights_only=False)
train_x = train['x'].unsqueeze(1).float().cuda()
train_y = train['y'].unsqueeze(1).float().cuda()
test_x = test['x'].unsqueeze(1).float().cuda()
test_y = test['y'].unsqueeze(1).float().cuda()
losses = []
started = time.perf_counter()
for epoch in range(3):
    model.train()
    total = 0.0
    for offset in range(0, len(train_x), 16):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(train_x[offset:offset+16]) - train_y[offset:offset+16]) ** 2).mean()
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(train_x[offset:offset+16])
    losses.append(total / len(train_x))
model.eval()
with torch.no_grad():
    predictions = torch.cat([model(test_x[i:i+16]) for i in range(0, len(test_x), 16)])
torch.cuda.synchronize()
pred = predictions.cpu().numpy()
target = test_y.cpu().numpy()
relative = np.linalg.norm((pred-target).reshape((len(pred),-1)),axis=1) / np.linalg.norm(target.reshape((len(pred),-1)),axis=1)
np.savez_compressed(out/'test-predictions.npz', prediction=pred, target=target, relative_l2=relative)
np.save(out/'test-inputs.npy', test_x.cpu().numpy())
np.savez_compressed(out/'weights.npz', **{name:value.detach().cpu().numpy() for name,value in model.state_dict().items() if isinstance(value, torch.Tensor)})
result = {'status':'pilot_only', 'train_examples':len(train_x), 'test_examples':len(test_x),
          'epochs':3, 'batch_size':16, 'train_mse_by_epoch':losses,
          'mean_relative_l2':float(relative.mean()), 'wall_seconds':time.perf_counter()-started,
          'train_data_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
          'test_data_sha256':hashlib.sha256(test_path.read_bytes()).hexdigest()}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
