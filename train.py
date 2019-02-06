from model import Arty
from controller import Controller
import torch
from pathlib import Path


def clear_output_dir():
  for path in Path('output').iterdir():
    path.unlink()

def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # device = 'cpu'

  clear_output_dir()

  # Create model
  model = Arty(device)
  controller = Controller(model)
  controller.train(device)

if __name__ == '__main__':
  main()