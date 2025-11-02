import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.functional import nll_loss
from torch.optim import Adam

import utils_nn as utils


model_args = {}
model_args['batch_size'] = 64
# learning rate is how fast it will descend
model_args['lr'] = 1.e-4 # 3.e-3
# the number of epochs is the number of times you go through the full dataset 
model_args['epochs'] = 20
# speedup the traninig
model_args['momentum'] = .8
# L@ regularization
model_args['weight_decay'] = 1.e-3
# get device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
     

train_loader, val_loader= utils.parse_dataset_classification('./generated', 
                                                             splitratio=[.7,.3],
                                                             batchsize=model_args['batch_size'], 
                                                             normalize=False, 
                                                             downsample=True)

#model = utils.Deep(no_classes=17).to(device)
#model = utils.SimpleCNN(no_classes=17).to(device)
model = utils.Custom_VGG(pretrained=True, ipt_size=[250,250], vgg_type='vgg11', no_classes=17).to(device)

#criterion = nn.TripletMarginLoss(m = 0.1)
# optimizer = optim.SGD(model.parameters(), 
#                       lr=model_args['lr'], 
#                       momentum=model_args['momentum'],
#                       weight_decay=model_args['weight_decay'])
optimizer = Adam(model.parameters(), 
                 lr=model_args['lr'],
                 weight_decay=model_args['weight_decay'])
experiment = utils.Experiments(device, model, train_loader, val_loader, nll_loss(reduction='mean'), optimizer)

train_loss, val_loss, best_accuracy = experiment.run_training(model_args['epochs'])
print('\nTest accuracy for best epoch: {:.0f}%\n'.format(100.*best_accuracy))
utils.plot_loss(train_loss, val_loss, 'CNN model')

confusion_matrix = experiment.calculate_confusion()
utils.plot_confusion(confusion_matrix)