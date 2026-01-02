import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from movinets import MoViNet
from movinets.config import _C

from BagClipsDataset import BagClipsDataset, imshow_bags
from pathlib import Path

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Train violence detection model using Multiple Instance Learning"
    )

    parser.add_argument(
        "--path_positive_clips",
        type=str,
        help="Path to positive video clips (contain the target event)"
    )

    parser.add_argument(
        "--path_negative_clips",
        type=str,
        help="Path to negative video clips (do not contain the target event)"
    )


    parser.add_argument(
        "--folder_backbone_model",
        type=str,
        default="./movinet_weights",
        help="Path to pretrained backbone model weights"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save model every N epochs"
    )

    parser.add_argument(
        "--length_window",
        type=int,
        default=8,
        help="Number of consecutive frames per instance (window size)"
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride between consecutive windows when sampling frames"
    )

    parser.add_argument(
        "--folder_trained_models",
        type=str,
        help="Directory where models and training logs will be saved"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )

    return parser.parse_args()


def train_loop(dataloader, model, model_movinet, optimizer, device, list_loss):
    model.train()
    model_movinet.clean_activation_buffers()
    
    #######print loss##########
    num_batches = len(dataloader)
    max_prints = 10
    step = max(1, num_batches // max_prints)
    ############################

    for batch, data in enumerate(dataloader):

        x_tmp = data[0]
        sz_batch = x_tmp.shape[0]
        x = x_tmp.reshape(sz_batch * x_tmp.shape[1], *x_tmp.shape[2:])
        #x = torch.squeeze(data[0], axis = 0)
        x = x.to(torch.float32)
        x = x /255.0

        x = x.to(device)
        xf = model_movinet(x)
        norm_xf = F.normalize(xf, dim=1, p =2)
        y = model(norm_xf)

        loss = 0
        sz_yi = y.shape[0]//sz_batch
        for i in range(sz_batch):
            y_i = y[i*sz_yi:(i+1)*sz_yi]

            sz_y = y_i.shape[0]
            yav = y_i[0:sz_y//2]
            ya = torch.max(yav)
            ynv = y_i[sz_y//2:]
            yn = torch.max(ynv)
            loss_i = torch.max(torch.zeros(1, dtype=torch.float32, device=device), 1-ya+yn) + 0.00008 * torch.sum(yav) + 0.00008 * torch.norm(torch.sub(yav[0:-1], yav[1:]))
            loss = loss + (1/sz_batch)*loss_i
            
        model_movinet.clean_activation_buffers()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        #print("______________________________________")
        #print("loss:",loss)
        #print("______________________________________")
        
        list_loss.append(loss.item())
        if batch % step == 0 or batch == num_batches - 1:
           print(loss.item())


#classifier
class Net(nn.Module):
      def __init__(self):
         super(Net, self).__init__()
         self.linear_relu_stack = nn.Sequential(
         nn.Linear(2048, 512),
         nn.ReLU(),
         #nn.Dropout(0.6),
         nn.Linear(512, 32),
         nn.ReLU(),
         #nn.Dropout(0.6),
         nn.Linear(32, 1),
         #nn.Dropout(0.6),
         nn.Sigmoid(),
         )
         

      def forward(self, x):
          return self.linear_relu_stack(x)





args = get_args()

path_positive_clips = args.path_positive_clips
path_negative_clips = args.path_negative_clips
folder_backbone_model = args.folder_backbone_model
batch_size = args.batch_size
epochs = args.epochs
checkpoint_interval = args.checkpoint_interval
length_window = args.length_window
stride = args.stride
folder_trained_models = args.folder_trained_models


'''
path_positive_clips = './clips/Violence/positive'
path_negative_clips = './clips/Violence/negative'
epochs = 50 
checkpoint_interval = 1
length_window = 8 
stride = 4 

folder_trained_models = './results/training/weights/Violence'
'''

bagClipsDataset = BagClipsDataset(path_positive_clips, path_negative_clips, length_window, stride, (185, 185), (172, 172), True)
train_dataloader = DataLoader(bagClipsDataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net = Net()
net.to(device)


#folder_backbone_model = './movinet_weights'
model_movinet = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True, model_dir=folder_backbone_model)
model_movinet.classifier[3] = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
model_movinet.eval()

if device.type != "cpu":
   model_movinet.cuda()

for param in model_movinet.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adagrad(net.parameters(), lr=0.001, weight_decay=0.001)


#creating folder to store weights
Path(folder_trained_models).mkdir(parents=True, exist_ok=True)
for t in range(epochs):
    print("Epoch: ", t)
    list_loss = []
    train_loop(train_dataloader, net, model_movinet, optimizer, device, list_loss)
     
    if(((t+1) % checkpoint_interval == 0) or (t + 1 == epochs)):
      file_trained_model = os.path.join(folder_trained_models, 'model_' + str(t+1) + '.pt')
      file_list_loss = os.path.join(folder_trained_models, 'loss_' + str(t+1) + '.txt')
      print(file_trained_model)
      print(file_list_loss)
      torch.save(net, file_trained_model)

      with open(file_list_loss, 'w') as f:
           for item in list_loss:
               f.write("%s\n" % item)
