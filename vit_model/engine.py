"""
Contains model code to training model
"""
import torch
torch.manual_seed(42)
from tqdm.auto import tqdm

def train_step(model, data, loss_fn, optimizer, device):
  """
  Train model on a training dataset and returns training loss and accuracy  for a single epoch

  Args:
    model: torch.nn.Module - Model that trains
    data: torch.utils.data.DataLoader - Train dataset
    loss_fn: torch.nn.Module - Loss function
    optimizer: torch.optim.Optimizer - Optimizer function
    device: torch.device - Which device would be used (cuda or cpu)

  Returns:
   A tuple (train_loss, train_acc) with training loss and training accuracy

  """
  model.to(device)
  model.train()
  train_loss, train_acc = 0, 0
  for X, y in data:
    X, y = X.to(device), y.to(device)
    y_logits= model(X)
    y_preds = torch.softmax(y_logits, axis=-1).argmax(axis=-1)

    loss = loss_fn(y_logits, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    train_loss += loss
    train_acc += (y_preds == y).sum().item()/len(y_preds)


  train_loss /= len(data)
  train_acc /= len(data)

  return train_loss, train_acc


def test_step(model, data, loss_fn, device):
  """
  Give to model the testing dataset and returns testing loss and accuracy for a single epoch

  Args:
    model: torch.nn.Module - Model to be tested
    data: torch.utils.data.DataLoader - Test dataset
    loss_fn: torch.nn.Module - Loss function
    device: torch.device - Which device would be used (cuda or cpu)

  Returns:
   A tuple (test_loss, test_acc) with testing loss and testing accuracy

  """
  model.to(device)
  model.eval()
  test_loss, test_acc = 0, 0
  with torch.inference_mode():
    for X, y in data:
      X, y = X.to(device), y.to(device)

      y_logits= model(X)
      y_preds = torch.softmax(y_logits, axis=1).argmax(axis=1)

      test_loss += loss_fn(y_logits, y)
      test_acc += (y_preds == y).sum().item()/len(y_preds)

    test_loss /= len(data)
    test_acc /= len(data)

  return test_loss, test_acc


def train(model, train_data, test_data, loss_fn, optimizer, epochs, device):

  """

  Args:
    model: torch.nn.Module - Model to be trained and tested
    train_data: torch.utils.data.DataLoader - Train dataset
    test_data: torch.utils.data.DataLoader - Test dataset
    loss_fn: torch.nn.Module - Loss function
    optimizer: torch.optim.Optimizer - Optimizer function
    epochs: int - Number of epochs to training model
    device: torch.device - Which device would be used (cuda or cpu)

  Returns:
   A dictionary with training and testing losses and training and testing accuracy
   Each metrics has a value in a list for each epoch

   in the form:{'train_loss': [...],
                'train_acc': [...],
                'test_loss': [...],
                'test_acc': [...]
                }

  """
  # Create a dictionary with results
  results = {'train_loss': [],
             'train_acc': [],
             'test_loss': [],
             'test_acc': []
             }

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       data=train_data,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device
                                       )
    test_loss, test_acc = test_step(model=model,
                                    data=test_data,
                                    loss_fn=loss_fn,
                                    device=device)


    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

    # Print out results
    print(f'Epoch: {epoch+1} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}')
    print('\n')

  return results
