import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import pathlib
import os
import torchvision
from torch.utils.data import DataLoader
import models
import data_loading
import matplotlib.pyplot as plt
import training
import re

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

image_files = data_loading.get_files(path, extensions=['.jpg'])
tabular_files = data_loading.get_files(path, extensions=['.csv'])
print(len(image_files))
print(len(tabular_files))

renamed_files = 0
for path_image in image_files:
    new_path = re.sub(r'output_map(\d).jpg', r'output_map0\1.jpg', str(path_image))
    if new_path != str(path_image):
        os.rename(path_image, new_path)
        renamed_files += 1

if renamed_files > 0:
    image_files = data_loading.get_files(path, extensions=['.jpg'])
    tabular_files = data_loading.get_files(path, extensions=['.csv'])
    print(len(image_files))
    print(len(tabular_files))

print(image_files[0])
print(data_loading.fileLabeller(image_files[0]))

columns = ["t_1", "t_2", "t_3", "t_4", "t_5", "ct_1", "ct_2", "ct_3", "ct_4", "ct_5",
           "t_1_blindtime", "t_2_blindtime", "t_3_blindtime", "t_4_blindtime", "t_5_blindtime",
           "ct_1_blindtime", "ct_2_blindtime", "ct_3_blindtime", "ct_4_blindtime", "ct_5_blindtime",
           "t_1_mainweapon", "t_1_secweapon", "t_1_flashbangs", "t_1_hassmoke", "t_1_hasmolotov", "t_1_hashe",
           "t_1_armor", "t_1_hashelmet", "t_1_hasc4",
           "t_2_mainweapon", "t_2_secweapon", "t_2_flashbangs", "t_2_hassmoke", "t_2_hasmolotov", "t_2_hashe",
           "t_2_armor", "t_2_hashelmet", "t_2_hasc4",
           "t_3_mainweapon", "t_3_secweapon", "t_3_flashbangs", "t_3_hassmoke", "t_3_hasmolotov", "t_3_hashe",
           "t_3_armor", "t_3_hashelmet", "t_3_hasc4",
           "t_4_mainweapon", "t_4_secweapon", "t_4_flashbangs", "t_4_hassmoke", "t_4_hasmolotov", "t_4_hashe",
           "t_4_armor", "t_4_hashelmet", "t_4_hasc4",
           "t_5_mainweapon", "t_5_secweapon", "t_5_flashbangs", "t_5_hassmoke", "t_5_hasmolotov", "t_5_hashe",
           "t_5_armor", "t_5_hashelmet", "t_5_hasc4",
           "ct_1_mainweapon", "ct_1_secweapon", "ct_1_flashbangs", "ct_1_hassmoke", "ct_1_hasmolotov", "ct_1_hashe",
           "ct_1_armor", "ct_1_hashelmet", "ct_1_hasdefusekit",
           "ct_2_mainweapon", "ct_2_secweapon", "ct_2_flashbangs", "ct_2_hassmoke", "ct_2_hasmolotov", "ct_2_hashe",
           "ct_2_armor", "ct_2_hashelmet", "ct_2_hasdefusekit",
           "ct_3_mainweapon", "ct_3_secweapon", "ct_3_flashbangs", "ct_3_hassmoke", "ct_3_hasmolotov", "ct_3_hashe",
           "ct_3_armor", "ct_3_hashelmet", "ct_3_hasdefusekit",
           "ct_4_mainweapon", "ct_4_secweapon", "ct_4_flashbangs", "ct_4_hassmoke", "ct_4_hasmolotov", "ct_4_hashe",
           "ct_4_armor", "ct_4_hashelmet", "ct_4_hasdefusekit",
           "ct_5_mainweapon", "ct_5_secweapon", "ct_5_flashbangs", "ct_5_hassmoke", "ct_5_hasmolotov", "ct_5_hashe",
           "ct_5_armor", "ct_5_hashelmet", "ct_5_hasdefusekit",
           "round_time",
           'related_image', 'winner']

full_csv = data_loading.filterTabularData(tabular_files, columns)
filtered_image_files = data_loading.filterImageData(image_files, full_csv)

splits = data_loading.roundSplitter(filtered_image_files)

print(filtered_image_files[55])
full_csv.iloc[55, :]

# class groups:
# mainweapon, secweapon,flashbangs,hassmoke,hasmolotov,hashe,hashelmet,hasc4,hasdefusekit

x_category_map = {"mainweapon": [0, 101, 102, 103, 104, 105, 106, 107,
                                 201, 202, 203, 204, 205, 206,
                                 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311],
                  "secweapon": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "flashbangs": [0, 1, 2],
                  "hassmoke": [0, 1],
                  "hasmolotov": [0, 1],
                  "hashe": [0, 1],
                  "hashelmet": [0, 1],
                  "hasc4": [0, 1],
                  "hasdefusekit": [0, 1]}

y_category_map = {"winner": ["t", "ct"]}

train_tabular = full_csv.iloc[splits[0], :]
train_rounds, train_winners = data_loading.get_rounds_and_winners(train_tabular)
train_images = [image for i, image in enumerate(filtered_image_files) if i in splits[0]]

valid_tabular = full_csv.iloc[splits[1], :]
valid_rounds, valid_winners = data_loading.get_rounds_and_winners(valid_tabular)
valid_images = [image for i, image in enumerate(filtered_image_files) if i in splits[1]]

cat_transforms = data_loading.TransformPipeline([data_loading.Categorize(x_category_map), data_loading.ToTensor])
cont_transforms = data_loading.TransformPipeline(
    [data_loading.Normalize(train_tabular.iloc[:, :-2], category_map=x_category_map), data_loading.ToTensor])
label_transforms = data_loading.TransformPipeline(
    [data_loading.Categorize(y_category_map, multicat=False), data_loading.ToTensor])
image_transforms = torchvision.transforms.Compose(
    # [torchvision.transforms.Resize(200), torchvision.transforms.ToTensor()])
    [torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()])
batch_transform = data_loading.NormalizeImage((0.0019, 0.0040, 0.0061), (0.0043, 0.0084, 0.0124))

train_x_tabular = train_tabular.iloc[:, :-2]
train_dataset = data_loading.CSGORoundsDataset(train_images, train_rounds, train_x_tabular, train_winners,
                                               image_transforms,
                                               cat_transforms, cont_transforms, label_transforms, x_category_map,
                                               y_category_map, batch_transform)

valid_y_dict = valid_tabular["winner"]
valid_x_tabular = valid_tabular.iloc[:, :-2]
valid_dataset = data_loading.CSGORoundsDataset(valid_images, valid_rounds, valid_x_tabular, valid_winners,
                                               image_transforms,
                                               cat_transforms, cont_transforms, label_transforms, x_category_map,
                                               y_category_map, batch_transform)

train_dl_mixed = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6,
                                             collate_fn=data_loading.pad_snapshot_sequence(seq_size), drop_last=True)
valid_dl_mixed = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=6,
                                             collate_fn=data_loading.pad_snapshot_sequence(seq_size), drop_last=True)

# tensor_to_image = torchvision.transforms.ToPILImage(mode="RGB")
# train_example_batch = batch_transform.decode(next(iter(train_dl_mixed)))
# mask_size=torch.sum((train_example_batch[3]==True),dim=1)
# key_image = train_example_batch[2][:, mask_size,:, :, :]

cont_names = ['t_1', 't_2', 't_3', 't_4', 't_5',
              'ct_1', 'ct_2', 'ct_3', 'ct_4', 'ct_5',
              "t_1_blindtime", "t_2_blindtime", "t_3_blindtime", "t_4_blindtime", "t_5_blindtime",
              "ct_1_blindtime", "ct_2_blindtime", "ct_3_blindtime", "ct_4_blindtime", "ct_5_blindtime",
              "t_1_armor", "t_2_armor", "t_3_armor", "t_4_armor", "t_5_armor",
              "ct_1_armor", "ct_2_armor", "ct_3_armor", "ct_4_armor", "ct_5_armor", "round_time"]
cat_names = ["t_1_mainweapon", "t_1_secweapon", "t_1_flashbangs", "t_1_hassmoke", "t_1_hasmolotov", "t_1_hashe",
             "t_1_hashelmet", "t_1_hasc4",
             "t_2_mainweapon", "t_2_secweapon", "t_2_flashbangs", "t_2_hassmoke", "t_2_hasmolotov", "t_2_hashe",
             "t_2_hashelmet", "t_2_hasc4",
             "t_3_mainweapon", "t_3_secweapon", "t_3_flashbangs", "t_3_hassmoke", "t_3_hasmolotov", "t_3_hashe",
             "t_3_hashelmet", "t_3_hasc4",
             "t_4_mainweapon", "t_4_secweapon", "t_4_flashbangs", "t_4_hassmoke", "t_4_hasmolotov", "t_4_hashe",
             "t_4_hashelmet", "t_4_hasc4",
             "t_5_mainweapon", "t_5_secweapon", "t_5_flashbangs", "t_5_hassmoke", "t_5_hasmolotov", "t_5_hashe",
             "t_5_hashelmet", "t_5_hasc4",
             "ct_1_mainweapon", "ct_1_secweapon", "ct_1_flashbangs", "ct_1_hassmoke", "ct_1_hasmolotov", "ct_1_hashe",
             "ct_1_hashelmet", "ct_1_hasdefusekit",
             "ct_2_mainweapon", "ct_2_secweapon", "ct_2_flashbangs", "ct_2_hassmoke", "ct_2_hasmolotov", "ct_2_hashe",
             "ct_2_hashelmet", "ct_2_hasdefusekit",
             "ct_3_mainweapon", "ct_3_secweapon", "ct_3_flashbangs", "ct_3_hassmoke", "ct_3_hasmolotov", "ct_3_hashe",
             "ct_3_hashelmet", "ct_3_hasdefusekit",
             "ct_4_mainweapon", "ct_4_secweapon", "ct_4_flashbangs", "ct_4_hassmoke", "ct_4_hasmolotov", "ct_4_hashe",
             "ct_4_hashelmet", "ct_4_hasdefusekit",
             "ct_5_mainweapon", "ct_5_secweapon", "ct_5_flashbangs", "ct_5_hassmoke", "ct_5_hasmolotov", "ct_5_hashe",
             "ct_5_hashelmet", "ct_5_hasdefusekit"]


group_count = 0
category_groups_sizes = {}
for class_group, class_group_categories in x_category_map.items():
    category_groups_sizes[class_group] = (group_count, len(class_group_categories))
    group_count += 1
category_list = [category for category in x_category_map]
tab_model = models.TabularModelCustom(category_list, category_groups_sizes, len(cont_names), [200, 100], ps=[0.2, 0.2])

image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=False, num_classes=image_output_size)
image_model.to("cuda:0")

# configuration = BertConfig(hidden_size=image_output_size+tabular_output_size,
#                            num_attention_heads=num_attention_heads,intermediate_size=intermediate_size,hidden_dropout_prob=0.2,
#                                  attention_probs_dropout_prob=0.2,output_hidden_states=True)
#
# class_model = BertModel(configuration)

encoder_layer = torch.nn.TransformerEncoderLayer(d_model=image_output_size + tabular_output_size, nhead=4)
seq_model = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
# configuration = DistilBertConfig(dim=tabular_output_size+image_output_size,
#                            n_heads=num_attention_heads,hidden_dim=intermediate_size,dropout=0.2,sinusoidal_pos_embds=True,
#                                  attention_dropout=0.2)
#
# class_model = DistilBertForSequenceClassification(configuration)

seq_model.to("cuda:0")

image_model.load_state_dict(
    torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/image_output-200.pt')))
tab_model.load_state_dict(
    torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/tabular_output-200.pt')))

model = models.CustomMixedModel(image_model, tab_model, seq_model)
model = model.to("cuda:0")


loss_fn = torch.nn.CrossEntropyLoss().cuda()

lr = 5e-4
n_epochs = 4

now = datetime.datetime.now()
creation_time = now.strftime("%H:%M")
tensorboard_writer = SummaryWriter(os.path.expanduser('~/projetos/data/csgo_analyze/experiment/tensorboard/') +
                                   now.strftime("%Y-%m-%d-") + creation_time)

tensorboard_class = training.TensorboardClass(tensorboard_writer)
# torch.set_num_threads(8)
optimizer = torch.optim.Adam(model.parameters(), lr=lr  # ,betas=(0.6,0.99),weight_decay=1e-4
                             )

lmbda = lambda epoch: 0.2
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
for epoch in range(n_epochs):
    training.one_epoch(loss_fn, model, train_dl_mixed, valid_dl_mixed, optimizer, tensorboard_class)
    scheduler.step()

torch.save(model.seq_model.state_dict(), str(pathlib.Path(path) / "seq_model.pt"))
