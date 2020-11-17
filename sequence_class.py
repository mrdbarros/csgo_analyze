import random

import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import pathlib
import os
import torchvision
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertForSequenceClassification, BertConfig, BertModel
import numpy as np
import models
import data_loading
import matplotlib.pyplot as plt
import training
import re
import logging


os.environ["TOKENIZERS_PARALLELISM"] = "false"
seq_size = 50
bs = 32
intermediate_size = 200
tabular_output_size = 200
image_output_size = 200
num_attention_heads = 10
img_size = 200
path = "/home/marcel/projetos/data/csgo_analyze/processed_test"
embeds_size = image_output_size + tabular_output_size
experiment_name="model_training"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed=421
np.random.seed(seed)
torch.manual_seed(seed)
device = "cuda:0"
logging.basicConfig(filename='sequence_class.log', level=logging.INFO)
max_image_batch=200
image_files = data_loading.get_files(path, extensions=['.jpg'])
tabular_files = data_loading.get_files(path, extensions=['.csv'])
model_start = 8
num_workers=6
cut_size = 0
if cut_size != 0:
    randomlist = random.sample(range(len(tabular_files)), cut_size)
    tabular_files = [tab_file for i,tab_file in enumerate(tabular_files) if i in randomlist]


# checks for invalid winner and rounds without tabular or winner files
full_csv = data_loading.filterTabularData(tabular_files, data_loading.columns)


filtered_image_files = data_loading.filterImageData(full_csv)


splits = data_loading.folderSplitter(filtered_image_files)

assert filtered_image_files[55]==full_csv.iloc[55, -2]

# class groups:
# mainweapon, secweapon,flashbangs,hassmoke,hasmolotov,hashe,hashelmet,hasc4,hasdefusekit



train_tabular = full_csv.iloc[splits[0], :]
train_rounds, train_winners = data_loading.get_rounds_and_winners(train_tabular)
train_images = [image for i, image in enumerate(filtered_image_files) if image.rfind("/train")!=-1]

valid_tabular = full_csv.iloc[splits[1], :]
valid_rounds, valid_winners = data_loading.get_rounds_and_winners(valid_tabular)
valid_images = [image for i, image in enumerate(filtered_image_files) if image.rfind("/val")!=-1]

cat_transforms = data_loading.TransformPipeline([data_loading.Categorize(data_loading.x_category_map), data_loading.ToTensor])
cont_transforms = data_loading.TransformPipeline(
    [data_loading.Normalize(train_tabular.iloc[:, :-2], category_map=data_loading.x_category_map), data_loading.ToTensor])
label_transforms = data_loading.TransformPipeline(
    [data_loading.Categorize(data_loading.y_category_map, multicat=False), data_loading.ToTensor])
image_transforms = torchvision.transforms.Compose(
    # [torchvision.transforms.Resize(200), torchvision.transforms.ToTensor()])
    [data_loading.resize_lycon(img_size)])
batch_transform = data_loading.TransformPipeline([data_loading.ToTensor, data_loading.NormalizeImage((0.0019, 0.0040, 0.0061),
                                                                                                     (0.0043, 0.0084, 0.0124))])

train_x_tabular = train_tabular.iloc[:, :-2]
#
# dali_iterator = data_loading.ExternalInputIterator( train_images, train_rounds, train_x_tabular, train_winners,bs, cat_transforms,
#                  cont_transforms, label_transforms, data_loading.x_category_map, data_loading.y_category_map)
#
#
# pipe = data_loading.ExternalSourcePipeline(image_size = img_size,batch_size=bs, num_threads=2, device_id = 0,
#                               external_data = dali_iterator,seed=seed)
# pii = data_loading.PyTorchIterator(pipe, size=dali_iterator.size, last_batch_padded=True, fill_last_batch=False)



train_dataset = data_loading.CSGORoundsDataset(train_images, train_rounds, train_x_tabular, train_winners,seq_size,
                                               image_transforms,
                                               cat_transforms, cont_transforms, label_transforms, data_loading.x_category_map,
                                               data_loading.y_category_map, batch_transform)





valid_y_dict = valid_tabular["winner"]
valid_x_tabular = valid_tabular.iloc[:, :-2]
valid_dataset = data_loading.CSGORoundsDataset(valid_images, valid_rounds, valid_x_tabular, valid_winners,seq_size,
                                               image_transforms,
                                               cat_transforms, cont_transforms, label_transforms, data_loading.x_category_map,
                                               data_loading.y_category_map, batch_transform)



train_dl_mixed = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers,
                                             collate_fn=data_loading.snapshot_sequence_batching(seq_size), drop_last=True)
valid_dl_mixed = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=num_workers,
                                             collate_fn=data_loading.snapshot_sequence_batching(seq_size), drop_last=True)

# tensor_to_image = torchvision.transforms.ToPILImage(mode="RGB")
# train_example_batch = batch_transform.decode(next(iter(train_dl_mixed)))
# mask_size=torch.sum((train_example_batch[3]==True),dim=1)
# key_image = train_example_batch[2][:, mask_size,:, :, :]


prepare_and_pad = data_loading.prepare_and_pad_sequence(seq_size)

group_count = 0
category_groups_sizes = {}
for class_group, class_group_categories in data_loading.x_category_map.items():
    category_groups_sizes[class_group] = (group_count, len(class_group_categories))
    group_count += 1
category_list = data_loading.cat_names


lr = 5e-6
n_epochs = 4
now = datetime.datetime.now()
creation_time = now.strftime("%H:%M")
tensorboard_writer = SummaryWriter(os.path.expanduser('~/projetos/data/csgo_analyze/experiment/tensorboard/sequence_class/') +
                                   experiment_name+"/"+now.strftime("%Y-%m-%d")+"/"+creation_time +"/"+
                                   "seed-"+str(seed)+"-lr-"+str(lr))

tensorboard_class = training.TensorboardClass(tensorboard_writer)

tab_model = models.TabularModelCustom(category_list, category_groups_sizes, len(data_loading.cont_names), [200, 100], ps=[0.1, 0.1])

image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=False, num_classes=image_output_size)
image_model.to(device)

configuration = BertConfig(hidden_size=image_output_size+tabular_output_size,
                           num_attention_heads=num_attention_heads,intermediate_size=intermediate_size,output_hidden_states=True)

seq_model = BertModel(configuration)

# encoder_layer = torch.nn.TransformerEncoderLayer(d_model=image_output_size + tabular_output_size, nhead=4)
# seq_model = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
# configuration = DistilBertConfig(dim=tabular_output_size+image_output_size,
#                            n_heads=num_attention_heads,hidden_dim=intermediate_size,dropout=0.2,sinusoidal_pos_embds=True,
#                                  attention_dropout=0.2)

# seq_model = DistilBertForSequenceClassification(configuration)

seq_model.to(device)

# image_model.load_state_dict(
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/image_output-bceloss200-1.pt')))
# tab_model.load_state_dict(
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/tab_output-bceloss200-1.pt')))


model = models.CustomMixedModel(image_model, tab_model, seq_model,image_output_size, embeds_size,prepare_and_pad,max_image_batch)
model = model.to(device)
model.load_state_dict(torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/full_model-unfrozen-seq_model'+ str(model_start)+".pt")))
# model.load_state_dict(
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/full_model-0.pt')))

loss_fn = torch.nn.BCEWithLogitsLoss().to(device=device)





#torch.set_num_threads(5)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr  # ,betas=(0.6,0.99),weight_decay=1e-4
                             )

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl_mixed), epochs=n_epochs)
# for epoch in range(n_epochs):
#     print("epoch:",epoch)
#     training.one_epoch(loss_fn, model, train_dl_mixed, valid_dl_mixed, optimizer, tensorboard_class,scheduler,train_embeds=False,train_seq_model=False)
#     #torch.save(model.seq_model.state_dict(), str(pathlib.Path(path) / "seq_model-")+str(epoch)+".pt")
#     torch.save(model.state_dict(), str(pathlib.Path(path) / "full_model-")+str(epoch)+".pt")
#
# lr=lr/10.0
# optimizer = optimizer_type(model.parameters(), lr=lr  # ,betas=(0.6,0.99),weight_decay=1e-4
#      )


scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*10, steps_per_epoch=len(train_dl_mixed), epochs=n_epochs)
for epoch in range(n_epochs):
    model_start = model_start + 1
    logging.info('epoch: %s',epoch)
    training.one_epoch(loss_fn, model, train_dl_mixed, valid_dl_mixed, optimizer, tensorboard_class,scheduler,device,train_embeds=False,train_seq_model=True)
    torch.save(model.seq_model.state_dict(), str(pathlib.Path(path) / "seq_model-")+str(model_start)+".pt")
    torch.save(model.state_dict(), str(pathlib.Path(path) / "full_model-unfrozen-seq_model")+str(model_start)+".pt")

#torch.save(model.seq_model.state_dict(), str(pathlib.Path(path) / "seq_model.pt"))



