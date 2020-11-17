import logging

import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import pathlib
import os
import torchvision
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

import models
import data_loading
import matplotlib.pyplot as plt
import training
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
seq_size = 50
bs = 1
intermediate_size = 200
tabular_output_size = 200
image_output_size = 200
num_attention_heads = 10
img_size = 200
is_eval=True
path = "/home/marcel/projetos/data/csgo_analyze/processed_test/val"
embeds_size = image_output_size + tabular_output_size
model_start=12
image_files = data_loading.get_files(path, extensions=['.jpg'])
tabular_files = data_loading.get_files(path, extensions=['.csv'])
max_image_batch=200
device="cuda:0"
logging.basicConfig(filename='evaluation.log', level=logging.INFO)
full_csv = data_loading.filterTabularData(tabular_files, data_loading.columns)
filtered_image_files = data_loading.filterImageData(full_csv)


assert filtered_image_files[55]==full_csv.iloc[55, -2]



valid_tabular = full_csv
valid_rounds, valid_winners = data_loading.get_rounds_and_winners(valid_tabular)
valid_images = filtered_image_files

cat_transforms = data_loading.TransformPipeline([data_loading.Categorize(data_loading.x_category_map), data_loading.ToTensor])
cont_transforms = data_loading.TransformPipeline(
    [data_loading.Normalize(valid_tabular.iloc[:, :-2], category_map=data_loading.x_category_map), data_loading.ToTensor])
label_transforms = data_loading.TransformPipeline(
    [data_loading.Categorize(data_loading.y_category_map, multicat=False), data_loading.ToTensor])
image_transforms = torchvision.transforms.Compose(
    # [torchvision.transforms.Resize(200), torchvision.transforms.ToTensor()])
    [data_loading.resize_lycon(img_size)])
batch_transform = data_loading.TransformPipeline([data_loading.ToTensor, data_loading.NormalizeImage((0.0019, 0.0040, 0.0061),
                                                                                                     (0.0043, 0.0084, 0.0124))])



valid_y_dict = valid_tabular["winner"]
valid_x_tabular = valid_tabular.iloc[:, :-2]
valid_dataset = data_loading.CSGORoundsDataset(valid_images, valid_rounds, valid_x_tabular, valid_winners,seq_size,
                                               image_transforms,
                                               cat_transforms, cont_transforms, label_transforms, data_loading.x_category_map,
                                               data_loading.y_category_map, batch_transform,is_eval=is_eval)

valid_dl_mixed = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, #num_workers=6,
                                             collate_fn=data_loading.snapshot_sequence_batching(seq_size,is_eval=is_eval), drop_last=True)

prepare_and_pad = data_loading.prepare_and_pad_sequence(seq_size)
group_count = 0
category_groups_sizes = {}
for class_group, class_group_categories in data_loading.x_category_map.items():
    category_groups_sizes[class_group] = (group_count, len(class_group_categories))
    group_count += 1
category_list = data_loading.cat_names

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
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/image_output-bceloss200-0.pt')))
# tab_model.load_state_dict(
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/tab_output-bceloss200-0.pt')))


model = models.CustomMixedModel(image_model, tab_model, seq_model,image_output_size, embeds_size,prepare_and_pad,max_image_batch)
model = model.to(device)
model.load_state_dict(torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/full_model-unfrozen-seq_model'+ str(model_start)+".pt")))
# model.load_state_dict(
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/full_model-0.pt')))

loss_fn = torch.nn.BCEWithLogitsLoss().to(device=device)
now = datetime.datetime.now()
creation_time = now.strftime("%H:%M")
tensorboard_writer = SummaryWriter(os.path.expanduser('~/projetos/data/csgo_analyze/experiment/tensorboard/') +
                                   now.strftime("%Y-%m-%d-") + creation_time)

tensorboard_class = training.TensorboardClass(tensorboard_writer)
training.validation_cycle(valid_dl_mixed,model,loss_fn,tensorboard_class,device,train_embeds=False,train_seq_model=False)
