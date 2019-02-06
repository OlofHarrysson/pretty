import torch
import numpy as np
from PIL import Image
from model import Arty

class Controller(object):
  def __init__(self, model):
    super(Controller, self).__init__()
    self.model = model

  def train(self, device):
    model = self.model
    model.init_w()

    target_model = Arty(device)
    target_model.init_w()


    im_ind = 0
    n_interpolations = 3
    for _ in range(n_interpolations):
      for __ in range(400):
        self.gen_image(model, im_ind)
        model.update_weights(target_model, tau=0.005)
        im_ind += 1
        print(im_ind)

      target_model.init_w()


    # torch.save(model.state_dict(), 'w1')
    # torch.save(model.state_dict(), 'w2')

  def gen_image(self, model, ind):
    size = 1024
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    colors = np.zeros((size, size, 2))
    for i in x:
      for j in y:
        colors[i][j] = np.array([float(i)/size-0.5, float(j)/size-0.5])
    colors = colors.reshape(size*size, 2)
    inp = torch.tensor(colors).type(torch.FloatTensor)
    img = model(inp)


    img = img.cpu().detach().numpy()
    im = img.reshape(size, size, 3)
    im = (im * 255).astype(np.uint8)
    im = Image.fromarray(im)
    # im.show()
    im.save('output/image-{}.png'.format(ind))