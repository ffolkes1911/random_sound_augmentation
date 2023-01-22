import os
import sys
import pickle
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torchvision

_US8k = 'US8K'
_ESC50 = 'ESC50'
DATASET_PATH = 'Datasets'
RESULT_PATH = 'Results'

dataset = _US8k

if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
print(device)

USE_GOOGLE_COLAB = False
if USE_GOOGLE_COLAB:
    from google.colab import drive
    drive.mount('/content/gdrive')

    # change the current working directory
    DATASET_PATH = '/content/gdrive/My Drive/' + DATASET_PATH
    RESULT_PATH = '/content/gdrive/My Drive/' + RESULT_PATH

dataset_df = pd.read_pickle(f"{DATASET_PATH}/{dataset}_df.pkl")
INPUT_SIZE = dataset_df.iloc[0]['melspectrogram'].shape
RESHAPE_SIZE = (1, INPUT_SIZE[0], INPUT_SIZE[1])
print(f"Input size: {INPUT_SIZE}")


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, pair_type='same'):
        assert isinstance(dataframe, pd.DataFrame)
        assert len(dataframe.columns) == 3

        self.dataframe = dataframe
        self.transform = transform
        self.pair_type = pair_type

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image, label, text = self.dataframe.iloc[index]
        images = [image.copy()]
        labels = [label.copy()]

        # print('picking random image')
        # find pair for current image
        while True:
          index_rand = np.random.randint(len(self.dataframe), size=1)[0]
          rand_img, rand_label, _ = self.dataframe.iloc[index_rand]
          if (self.pair_type == 'same' and rand_label == label) or \
            (self.pair_type == 'diff' and rand_label != label):
            images.append(rand_img.copy())
            labels.append(rand_label.copy())
            break
            
        # print('transforming image')
        if self.transform is not None:
            image = self.transform(images)[0]

        return {'spectrogram': image, 'label': label, 'text': text }


class RightShift(object):
    """Shift the image to the right in time."""

    def __init__(self, width_shift_range, probability=1.0):
        assert isinstance(width_shift_range, (int, float))
        assert isinstance(probability, (float))

        if isinstance(width_shift_range, int):
            assert width_shift_range > 0
            self.width_shift_range = width_shift_range
        else:
            assert width_shift_range > 0.0
            assert width_shift_range <= 1.0
            self.width_shift_range = int(width_shift_range * self.input_size[1])
                        
        assert probability > 0.0 and probability <= 1.0
        self.probability = probability

    def __call__(self, images):
        if np.random.random() > self.probability:
          return images

        image = images[0]
        # create a new array filled with the min value
        shifted_image= np.full(image.shape, np.min(image), dtype='float32')

        # randomly choose a start postion
        rand_position = np.random.randint(1, self.width_shift_range)

        # shift the image
        shifted_image[:,rand_position:] = copy.deepcopy(image[:,:-rand_position])

        images[0] = shifted_image
        return images


class GaussNoise(object):
    """Add Gaussian noise to the spectrogram image."""

    def __init__(self, mean=0.0, std=None, probability=1.0):
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float)) or std is None
        assert isinstance(probability, (float))

        self.mean = mean

        if std is not None:
            assert std > 0.0
            self.std = std
        else:
            self.std = std

        assert probability > 0.0 and probability <= 1.0
        self.probability = probability

    def __call__(self, images):
      if np.random.random() > self.probability:
          return images

      image = images[0]
      # print('adding noise')
      # set some std value 
      min_pixel_value = np.min(image)
      if self.std is None:
        std_factor = 0.03     # factor number 
        std = np.abs(min_pixel_value*std_factor)

      # generate a white noise image
      gauss_mask = np.random.normal(self.mean, 
                                    std, 
                                    size=image.shape).astype('float32')
      
      # add white noise to the sound image
      noisy_image = image + gauss_mask
      images[0] = noisy_image

      return images


class Reshape(object):
    """Reshape the image array."""

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))

        self.output_size = output_size

    def __call__(self, images):
      # print(f"reshape from {images[0].shape}")
      images[0] = images[0].reshape(self.output_size)
      return images


# below methods based https://github.com/ceciliaresearch/MixedExample/blob/master/mixed_example.py
class Mixup(object):
    """merge whole images"""

    def __init__(self, mixup_min=0.4, mixup_max=0.6, probability=1.0):
        assert isinstance(mixup_min, (float))
        assert isinstance(mixup_max, (float))

        self.mixup_min = mixup_min
        self.mixup_max = mixup_max
        self.probability = probability

    def __call__(self, images):
      if np.random.random() > self.probability:
        return images

      mix_coeff = np.random.random() * (self.mixup_max - self.mixup_min) + self.mixup_min 


      for img in images[1:]:
        images[0] = mix_coeff * images[0] + (1 - mix_coeff) * img
      return images


class vert_mixup(object):
    """mixup images vertically"""

    def __init__(self, mixup_min=0.4, mixup_max=0.6, probability=1.0):
        assert isinstance(mixup_min, (float))
        assert isinstance(mixup_max, (float))

        self.mixup_min = mixup_min
        self.mixup_max = mixup_max
        self.probability = probability

    def __call__(self, images):
      if np.random.random() > self.probability:
        return images

      mix_coeff = np.random.random() * (self.mixup_max - self.mixup_min) + self.mixup_min 
      cut_point = int(len(images[0]) * mix_coeff)

      # images[0] = np.concatenate((images[0][:cut_point,:], images[1][cut_point:,:]),0)

      for img in images[1:]:
        images[0][cut_point:,:] = mix_coeff * images[0][cut_point:,:] + (1 - mix_coeff) * img[cut_point:,:]

      return images


class horiz_mixup(object):
    """mixup images horizontaly"""

    def __init__(self, mixup_min=0.4, mixup_max=0.6, p=0.5, probability=1.0):
        assert isinstance(mixup_min, (float))
        assert isinstance(mixup_max, (float))

        self.mixup_min = mixup_min
        self.mixup_max = mixup_max
        self.probability = probability

    def __call__(self, images):
      if np.random.random() > self.probability:
        return images

      mix_coeff = np.random.random() * (self.mixup_max - self.mixup_min) + self.mixup_min 
      cut_point = int(len(images[0]) * mix_coeff)

      # images[0] = np.concatenate((images[0][:,:cut_point], images[1][:,cut_point:]),1)

      for img in images[1:]:
        images[0][:,:cut_point] = mix_coeff * images[0][:,:cut_point] + (1 - mix_coeff) * img[:,:cut_point]

      return images


class random_2x2(object):
    """concat images horizontaly"""

    def __init__(self, mixup_min=0.4, mixup_max=0.6, p=0.5, probability=1.0):
        assert isinstance(mixup_min, (float))
        assert isinstance(mixup_max, (float))

        self.mixup_min = mixup_min
        self.mixup_max = mixup_max
        self.probability = probability
        self.p = p
        self.offset_min = 0.5 * p
        self.offset_max = 1 - self.offset_min

    def __call__(self, images):
      if np.random.random() > self.probability:
        return images

      (H, W) = images[0].shape

      mix_coeff = np.random.random() * (self.mixup_max - self.mixup_min) + self.mixup_min
      h_cutpoint = int(H * (np.random.random() * (self.offset_max - self.offset_min) + self.offset_min))
      w_cutpoint = int(W * (np.random.random() * (self.offset_max - self.offset_min) + self.offset_min))
      rand_idxs = np.random.randint(0, len(images), 4)

      images[0][:h_cutpoint,:w_cutpoint] = mix_coeff * images[0][:h_cutpoint,:w_cutpoint] + \
        (1 - mix_coeff) * images[rand_idxs[0]][:h_cutpoint,:w_cutpoint] # top left
      images[0][:h_cutpoint,w_cutpoint:] = mix_coeff * images[0][:h_cutpoint,w_cutpoint:] + \
        (1 - mix_coeff) * images[rand_idxs[1]][:h_cutpoint,w_cutpoint:] # top right
      images[0][h_cutpoint:,:w_cutpoint] = mix_coeff * images[0][h_cutpoint:,:w_cutpoint] + \
        (1 - mix_coeff) * images[rand_idxs[2]][h_cutpoint:,:w_cutpoint] # bot left
      images[0][h_cutpoint:,w_cutpoint:] = mix_coeff * images[0][h_cutpoint:,w_cutpoint:] + \
        (1 - mix_coeff) * images[rand_idxs[3]][h_cutpoint:,w_cutpoint:] # bot right
        

      return images


class random_column_interval(object):
    """mixup one image column region"""

    def __init__(self, mixup_min=0.4, mixup_max=0.6, p=0.5, probability=1.0):
        assert isinstance(mixup_min, (float))
        assert isinstance(mixup_max, (float))

        self.mixup_min = mixup_min
        self.mixup_max = mixup_max
        self.probability = probability

    def __call__(self, images):
      if np.random.random() > self.probability:
        return images

      (H, W) = images[0].shape
      mix_coeff = np.random.random() * (self.mixup_max - self.mixup_min) + self.mixup_min 
      idx1 = np.random.randint(0, W)
      idx2 = np.random.randint(idx1, W)

      for img in images[1:]:
        images[0][:,idx1:idx2] = mix_coeff * images[0][:,idx1:idx2] + (1 - mix_coeff) * img[:,idx1:idx2]

      return images


class random_row_interval(object):
    """mixup one image row region"""

    def __init__(self, mixup_min=0.4, mixup_max=0.6, p=0.5, probability=1.0):
        assert isinstance(mixup_min, (float))
        assert isinstance(mixup_max, (float))

        self.mixup_min = mixup_min
        self.mixup_max = mixup_max
        self.probability = probability

    def __call__(self, images):
      if np.random.random() > self.probability:
        return images

      (H, W) = images[0].shape
      mix_coeff = np.random.random() * (self.mixup_max - self.mixup_min) + self.mixup_min 
      idx1 = np.random.randint(0, H)
      idx2 = np.random.randint(idx1, H)


      for img in images[1:]:
        images[0][idx1:idx2,:] = mix_coeff * images[0][idx1:idx2,:] + (1 - mix_coeff) * img[idx1:idx2,:]

      return images


class random_cols(object):
    """mixup random image cols"""

    def __init__(self, mixup_min=0.4, mixup_max=0.6, p=0.5, probability=1.0):
        assert isinstance(mixup_min, (float))
        assert isinstance(mixup_max, (float))

        self.mixup_min = mixup_min
        self.mixup_max = mixup_max
        self.probability = probability
        self.p = p

    def __call__(self, images):
      if np.random.random() > self.probability:
        return images

      (H, W) = images[0].shape
      idxs = np.random.choice([0, 1], size=W, p=[1-self.p, self.p])
      mix_coeff = np.random.random() * (self.mixup_max - self.mixup_min) + self.mixup_min 
      
      for img in images[1:]:
        images[0] = (1-idxs) * images[0] + idxs * mix_coeff * images[0] + idxs * (1 - mix_coeff) * img

      return images


class random_rows(object):
    """mixup random image rows"""

    def __init__(self, mixup_min=0.4, mixup_max=0.6, p=0.5, probability=1.0):
        assert isinstance(mixup_min, (float))
        assert isinstance(mixup_max, (float))

        self.mixup_min = mixup_min
        self.mixup_max = mixup_max
        self.probability = probability
        self.p = p

    def __call__(self, images):
      if np.random.random() > self.probability:
        return images

      (H, W) = images[0].shape
      idxs = np.random.choice([0, 1], size=H, p=[1-self.p, self.p])
      # idxs = np.ones(W, dtype=float) * idxs
      mix_coeff = np.random.random() * (self.mixup_max - self.mixup_min) + self.mixup_min
      # print(idxs)

      for img in images[1:]:
        orig_img_T = np.transpose(images[0])
        images[0] = np.transpose((1-idxs) * orig_img_T + idxs * mix_coeff * orig_img_T + idxs * (1 - mix_coeff) * np.transpose(img))

      return images

def normalize_data(train_df, test_df):
  # compute the mean and std (pixel-wise)
  mean = train_df['melspectrogram'].mean()
  std = np.std(np.stack(train_df['melspectrogram']), axis=0)

  # normalize train set
  train_spectrograms = (np.stack(train_df['melspectrogram']) - mean) / std
  train_labels = train_df['label'].to_numpy()
  train_folds = train_df['text'].to_numpy()
  train_df = pd.DataFrame(zip(train_spectrograms, train_labels, train_folds), columns=['melspectrogram', 'label', 'text'])

  # normalize test set
  test_spectrograms = (np.stack(test_df['melspectrogram']) - mean) / std
  test_labels = test_df['label'].to_numpy()
  test_folds = test_df['text'].to_numpy()
  test_df = pd.DataFrame(zip(test_spectrograms, test_labels, test_folds), columns=['melspectrogram', 'label', 'text'])

  return train_df, test_df


def init_dataloader(train_df, test_df, train_transforms, test_transforms, batch_size, seed):
  # split the data
  train_df, test_df = train_test_split(dataset_df, test_size=0.25, random_state=seed)
  # normalize the data
  train_df, test_df = normalize_data(train_df, test_df)
  train_ds = CustomDataset(train_df, transform=train_transforms)
  test_ds = CustomDataset(test_df, transform=test_transforms)
  train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
  test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = False)

  return train_loader, test_loader


def lr_decay(optimizer, epoch, learning_rate):
  if epoch%10==0:
    new_lr = learning_rate / (10**(epoch//10))
    optimizer.param_groups[0]['lr'] = new_lr
    tqdm.write(f'Changed learning rate to {new_lr}')
  return optimizer


def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, lr, change_lr=None):
  train_losses = []
  valid_losses = []
  train_accuracy = []
  valid_accuracy = []

  for epoch in tqdm(range(1,epochs+1), desc="Epoch"):
    model.train()
    batch_losses=[]
    trace_y = []
    trace_yhat = []

    if change_lr:
      optimizer = change_lr(optimizer, epoch, lr)

    for i, data in enumerate(train_loader):
      x = data['spectrogram'].to(device)
      y = data['label'].to(device)
      optimizer.zero_grad()
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.long)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      loss.backward()
      batch_losses.append(loss.item())
      optimizer.step()
      trace_y.append(y.cpu().detach().numpy())
      trace_yhat.append(y_hat.cpu().detach().numpy())

    train_losses.append(np.mean(batch_losses))

    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
    train_accuracy.append(np.mean(accuracy))
    #print(f'Epoch - {epoch} Train-Loss : {np.mean(batch_losses)}')

    model.eval()
    batch_losses=[]
    trace_y = []
    trace_yhat = []

    for i, data in enumerate(valid_loader):
      x = data['spectrogram'].to(device)
      y = data['label'].to(device)
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.long)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      trace_y.append(y.cpu().detach().numpy())
      trace_yhat.append(y_hat.cpu().detach().numpy())      
      batch_losses.append(loss.item())

    valid_losses.append(np.mean(batch_losses))

    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
    valid_accuracy.append(accuracy)
    #print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
  return model, train_losses, valid_losses, train_accuracy, valid_accuracy


def get_confusion_matrix(model, label_names, data_loader):
    y_pred = []
    y_true = []
    if torch.cuda.is_available():
      device=torch.device('cuda:0')
    else:
      device=torch.device('cpu')
    model.eval()
    # don't calculate gradient
    with torch.no_grad():
      for i, data in enumerate(data_loader):
        x = data['spectrogram'].to(device)
        y = data['label'].to(device)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)
        y_hat = model(x)

        y_true.extend(y.cpu().detach().numpy())
        y_pred.extend(np.argmax(y_hat.cpu().detach().numpy(),1))      


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [name for name in label_names], columns =  [name for name in label_names])
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    
    return df_cm

if __name__ == "__main__":
  train_transform_names = []
  train_transform_sets = []

  train_transform_names.append('No augmentation')
  train_transform_sets.append(transforms.Compose([
    Reshape(output_size=RESHAPE_SIZE),
  ]))
  # for prob in [0.3, 0.5, 1.0]:
  for prob in [1.0]:
    # for a_min, a_max in [(0.2, 0.3), (0.45, 0.55), (0.7, 0.8)]:
    for a_min, a_max in [(0.7, 0.8)]:
      train_transform_names.append(f"Mixup {a_min}-{a_max} prob={prob}")
      train_transform_sets.append(transforms.Compose([
        Mixup(probability=prob, mixup_min=a_min, mixup_max=a_max),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

      train_transform_names.append(f"Vert. mixup {a_min}-{a_max} prob={prob}")
      train_transform_sets.append(transforms.Compose([
        vert_mixup(probability=prob, mixup_min=a_min, mixup_max=a_max),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

      train_transform_names.append(f"Horiz. mixup {a_min}-{a_max} prob={prob}")
      train_transform_sets.append(transforms.Compose([
        horiz_mixup(probability=prob, mixup_min=a_min, mixup_max=a_max),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

      train_transform_names.append(f"Gaussian noise prob={prob}")
      train_transform_sets.append(transforms.Compose([
        GaussNoise(probability=prob),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

      train_transform_names.append(f"Random 2x2 mixup {a_min}-{a_max} prob={prob}")
      train_transform_sets.append(transforms.Compose([
        random_2x2(probability=prob, mixup_min=a_min, mixup_max=a_max),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

      train_transform_names.append(f"Random column interval mixup {a_min}-{a_max} prob={prob}")
      train_transform_sets.append(transforms.Compose([
        random_column_interval(probability=prob, mixup_min=a_min, mixup_max=a_max),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

      train_transform_names.append(f"Random row interval mixup {a_min}-{a_max} prob={prob}")
      train_transform_sets.append(transforms.Compose([
        random_row_interval(probability=prob, mixup_min=a_min, mixup_max=a_max),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

      train_transform_names.append(f"Random rows mixup {a_min}-{a_max} prob={prob}")
      train_transform_sets.append(transforms.Compose([
        random_rows(probability=prob, mixup_min=a_min, mixup_max=a_max),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

      train_transform_names.append(f"Random cols mixup {a_min}-{a_max} prob={prob}")
      train_transform_sets.append(transforms.Compose([
        random_cols(probability=prob, mixup_min=a_min, mixup_max=a_max),
        Reshape(output_size=RESHAPE_SIZE),
      ]))

  test_transforms = transforms.Compose([Reshape(output_size=RESHAPE_SIZE)])

  run_count = 5
  epochs = 25

  classes = dataset_df.text.unique()
  trained_models_results = []

  train_path = f"{dataset}_training_results"
  print(f"Starting training")
  print(f"creating path '{RESULT_PATH}/{train_path}'\n")
  Path(f"{RESULT_PATH}/{train_path}").mkdir(parents=True, exist_ok=True)
  os.chdir(f"{RESULT_PATH}/{train_path}")

  #loop over different augment sets
  for config_name, train_transforms in zip(tqdm(train_transform_names, desc="Config"), train_transform_sets):
    tqdm.write(f"Training using config '{config_name}'")
    #train multiple times
    tqdm.write(f"Using augments: {train_transforms}")
    
    model_result = {
      "config_name": config_name,
      "train_loss": [],
      "train_acc": [],
      "valid_loss": [],
      "valid_acc": []
    }

    shuffle_split = StratifiedShuffleSplit(n_splits=run_count, test_size=0.25, random_state=42)
    for i_run, (train_index, test_index) in enumerate(tqdm(shuffle_split.split(np.zeros(len(dataset_df)), dataset_df['label']), total=run_count, desc="Run")):
      train_df = dataset_df.iloc[train_index]
      test_df = dataset_df.iloc[test_index]
      # normalize the data
      train_df, test_df = normalize_data(train_df, test_df)
      train_ds = CustomDataset(train_df, transform=train_transforms)
      test_ds = CustomDataset(test_df, transform=test_transforms)
      train_loader = DataLoader(train_ds, batch_size = 16, shuffle = True)
      test_loader = DataLoader(test_ds, batch_size = 16, shuffle = False)

      tqdm.write(f"Model training run: {i_run}")
      learning_rate = 2e-4
      torch.cuda.empty_cache()
      
      model = models.resnet18(weights=torchvision.models.ResNet18_Weights)
      model.conv1=nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size[0], stride=model.conv1.stride[0], padding=model.conv1.padding[0])
      model.fc = nn.Sequential(*[nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, len(classes))])
      net = model.to(device)

      optimizer = optim.Adam(net.parameters(), lr=learning_rate)
      loss_fn = nn.CrossEntropyLoss()
      trained_model, train_losses, valid_losses, train_accuracy, valid_accuracy = train(net, loss_fn, train_loader, test_loader, epochs, optimizer, learning_rate, lr_decay)

      df_single_run = pd.DataFrame({
          "train_losses": train_losses,
          "valid_losses": valid_losses,
          "train_accuracy": train_accuracy,
          "valid_accuracy": valid_accuracy
      })
      df_single_run.to_csv(f"results_{config_name}_{i_run}.csv")

      model_result["train_loss"].append(min(train_losses))
      model_result["train_acc"].append(max(train_accuracy))
      model_result["valid_loss"].append(min(valid_losses))
      model_result["valid_acc"].append(max(valid_accuracy))

      df_cm = get_confusion_matrix(trained_model, classes, test_loader)
      df_cm.to_csv(f"cm_{config_name}_{i_run}.csv")

    df = pd.DataFrame(model_result)
    df.to_csv(f"results_{config_name}_all_runs.csv")
    print(model_result)

