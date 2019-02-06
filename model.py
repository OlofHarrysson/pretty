import torch
import torch.nn as nn

class Arty(nn.Module):
  def __init__(self, device, act=nn.Tanh, num_neurons=16, num_layers=9):
    super(Arty, self).__init__()

    layers = [nn.Linear(2, num_neurons, bias=True), act()]
    for _ in range(num_layers - 1):
      layers += [nn.Linear(num_neurons, num_neurons, bias=False), act()]
    layers += [nn.Linear(num_neurons, 3, bias=False), nn.Sigmoid()]
    self.layers = nn.Sequential(*layers)

    self.device = device
    self = self.to(device)
    # self.init_w()
          
  def forward(self, x):
    x = x.to(self.device)
    return self.layers(x)

  def init_w(self):
    self.apply(init_normal)


  def update_weights(self, target, tau):
    for tar_w, this_w in zip(target.parameters(), self.parameters()):
      this_w.data.copy_(tau*tar_w.data + (1.0-tau)*this_w.data)




def init_normal(m):
  if type(m) == nn.Linear:
    nn.init.normal_(m.weight)


