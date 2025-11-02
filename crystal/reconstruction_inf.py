import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import utils_nn as utils

from utils_gen import wallpaper_symmetries, plane_symmety_names, wallpaper_reduced
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

model = utils.Autoencoder(num_input_channels=1, latent_dim = 512, if_classifer=True, no_classes=23, device=None)
model.load_state_dict(torch.load("AE_model.pth"))
model.to(device)
model.eval()

_, val_loader = utils.parse_dataset_classification('./generated', 
                                                    label_dict=wallpaper_symmetries, 
                                                    splitratio=[.7,.3],
                                                    batchsize=64, 
                                                    normalize=False,
                                                    downsample=True)
reps = []
classes = []
with torch.no_grad():
    for data, labels in val_loader:
        data = data.to(device)
        # calculate and sum up batch loss
        z = model.encoder(data)
        labels_tr = []
        for l in labels.numpy().astype(int):
            for i, v in enumerate(wallpaper_symmetries.values()):
                if (v == l).all(): 
                    labels_tr.append(i)
                    break
        reps.append(z.cpu().numpy())
        classes.append(labels_tr)
reps  = np.concatenate(reps)
classes = np.concatenate(classes)
classes = [wallpaper_reduced[c+1] for c in classes]

import umap
reducer = umap.UMAP(n_components=5, metric='braycurtis', min_dist=0.3, n_neighbors=30)
embedding = reducer.fit_transform(reps)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
embds = ax.scatter(embedding[:, -2], embedding[:, -1], c=classes, cmap='Spectral')
fig.colorbar(embds).set_ticks(np.arange(1,8))
fig.savefig('embedding.png', dpi=200)