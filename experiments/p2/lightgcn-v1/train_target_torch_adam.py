"""Controller reference with PyTorch's Adam epsilon placement."""

import jittor as jt

from train_target_reference import LightGCNReference


class TorchAdam(jt.optim.Adam):
    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        n = float(self.n_step)
        jt.flags.node_order = 1
        for group in self.param_groups:
            lr = group.get("lr", self.lr)
            eps = group.get("eps", self.eps)
            decay = group.get("weight_decay", self.weight_decay)
            beta1, beta2 = group.get("betas", self.betas)
            variance_correction = jt.sqrt(1 - beta2**n)
            step_size = lr * variance_correction / (1 - beta1**n)
            for param, grad, variance, mean in zip(
                group["params"], group["grads"], group["values"], group["m"]
            ):
                if param.is_stop_grad():
                    continue
                grad = param * decay + grad
                mean.update(beta1 * mean + (1 - beta1) * grad)
                variance.update(beta2 * variance + (1 - beta2) * grad * grad)
                param.update(param - mean * step_size / (jt.sqrt(variance) + eps * variance_correction))
        self.post_step()


def load_train_model() -> LightGCNReference:
    return LightGCNReference()


def build_optimizer(model: LightGCNReference) -> TorchAdam:
    parameters = model.parameters()
    for parameter in parameters:
        parameter.start_grad()
    optimizer = TorchAdam(parameters, lr=0.001)
    optimizer.param_groups[0]["lr"] = 0.001
    return optimizer


def train_step(model: LightGCNReference, optimizer: TorchAdam, users, positive_items, negative_items):
    loss = model.bpr_loss(users, positive_items, negative_items)
    optimizer.step(loss)
    return loss
