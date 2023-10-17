"""

"""
from pathlib import Path
import torch
import os

def save_model(model: torch.nn.Module,
               save_dir: str,
               model_name: str):
  """
  Save the model's weights and biases

  Args:
    model: torch.nn.Module - Model that should be saved
    save_dir: str - Directory to save the model
    model_name: str - A filename like 'model_name.pth'

  Usage example:
    save_model(model=model_0,
               save_dir='models',
               model_name=model_0.pth)

  """

  # Create save model path
  save_path = Path(save_dir)
  save_path.mkdir(parents=True, exist_ok=True)
  model_save_path = save_path / model_name

  # Save the model
  print(f'Model saving to {model_save_path}')
  torch.save(model.state_dict(), model_save_path)


