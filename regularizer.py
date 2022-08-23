import torch


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module, lamda=0.0):
        super().__init__()
        self.module = module
        self.lamda = lamda
        self.batch_idx = 0

    def init_ps(self, train_dataloader):
        if self.lamda != 0.0:
            self.module.eval()
            ps = []
            for inputs, targets in iter(train_dataloader):
                outputs = self.module(inputs)
                p = torch.zeros_like(outputs)
                ps.append(torch.nn.Parameter(p, requires_grad=True))
            self.ps = torch.nn.ParameterList(ps)
            self.module.train()

    def set_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def forward(self, x):
        x = self.module(x)
        if self.lamda != 0.0 and self.training:
            x = x + self.lamda * self.ps[self.batch_idx]
        return x