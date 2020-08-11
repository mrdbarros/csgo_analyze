import itertools
import random

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import pathlib
import os
import PIL
import torchvision
import numpy as np
import data_loading
import models
import training





path = "/home/marcel/projetos/data/csgo_analyze/processed_test"
image_files = data_loading.get_files(path, extensions=['.jpg'])
tabular_files = data_loading.get_files(path, extensions=['.csv'])
print(len(image_files))
print(len(tabular_files))
bs = 360
final_bn=True
image_output_size=200
seed = 42
experiment_name = "model_training"


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.manual_seed(seed)
cut_size = 0
n_epochs =15
lr=5e-4



if cut_size != 0:
    randomlist = random.sample(range(len(tabular_files)), cut_size)
    tabular_files = [tab_file for i,tab_file in enumerate(tabular_files) if i in randomlist]


print(image_files[0])
print(data_loading.fileLabeller(image_files[0]))


full_csv=data_loading.filterTabularData(tabular_files,data_loading.columns)

filtered_image_files = data_loading.filterImageData(full_csv)


# filtered_image_files=filtered_image_files[:2000]
# full_csv=full_csv.iloc[:2000,:]
splits = data_loading.folderSplitter(filtered_image_files)

assert filtered_image_files[203]==full_csv.iloc[203, -2]

# class groups:
# mainweapon, secweapon,flashbangs,hassmoke,hasmolotov,hashe,hashelmet,hasc4,hasdefusekit



def ToTensor(o):
    return torch.from_numpy(o)


class Categorize():
    def __init__(self, category_map: list = None, multicat=True):
        self.category_map = category_map
        self.multicat = multicat

    def __call__(self, df_row: pd.DataFrame):
        categories = []
        if self.multicat:
            for cat in df_row.index:
                group = cat[cat.rfind("_") + 1:]
                category = self.category_map[group].index(df_row[cat])
                categories.append(category)
        else:
            category = self.category_map["winner"].index(df_row)
            categories.append(category)
        return np.array(categories)


class Normalize():
    def __init__(self, tabular_df: pd.DataFrame, category_map):
        self.means = [tabular_df[column].mean() for column in tabular_df.columns
                      if not column[column.rfind("_") + 1:] in category_map]
        self.std = [tabular_df[column].std() for column in tabular_df.columns
                    if not column[column.rfind("_") + 1:] in category_map]

    def __call__(self, o: pd.DataFrame):
        ret = o.copy()
        for i, column in enumerate(ret.index):
            ret[column] = (o[column] - self.means[i]) / self.std[i]
        return ret.astype("float32").values


class TransformPipeline():
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, o):
        res = o
        for transform in self.transforms:
            res = transform(res)
        return res





train_tabular = full_csv.iloc[splits[0], :]
train_images = [image for i, image in enumerate(filtered_image_files) if i in splits[0]]
valid_tabular = full_csv.iloc[splits[1], :]
valid_images = [image for i, image in enumerate(filtered_image_files) if i in splits[1]]

cat_transforms = TransformPipeline([Categorize(data_loading.x_category_map), ToTensor])
cont_transforms = TransformPipeline([Normalize(train_tabular.iloc[:, :-2], category_map=data_loading.x_category_map), ToTensor])
label_transforms = TransformPipeline([Categorize(data_loading.y_category_map, multicat=False), ToTensor])
image_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(200), torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.0019, 0.0040, 0.0061), (0.0043, 0.0084, 0.0124))])
train_y_series = train_tabular["winner"]
train_x_tabular = train_tabular.iloc[:, :-2]
train_dataset = data_loading.CSGORoundsDatasetSingleImage(train_images, train_x_tabular, train_y_series, image_transforms,
                                  cat_transforms, cont_transforms, label_transforms, data_loading.x_category_map, data_loading.y_category_map)

valid_y_series = valid_tabular["winner"]
valid_x_tabular = valid_tabular.iloc[:, :-2]
valid_dataset = data_loading.CSGORoundsDatasetSingleImage(valid_images, valid_x_tabular, valid_y_series, image_transforms,
                                  cat_transforms, cont_transforms, label_transforms, data_loading.x_category_map, data_loading.y_category_map)

print(train_dataset[2])
train_dl_mixed = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6
                                             )
valid_dl_mixed = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=6
                                           )


group_count = 0
category_groups_sizes = {}
for class_group, class_group_categories in data_loading.x_category_map.items():
    category_groups_sizes[class_group] = (group_count, len(class_group_categories))
    group_count += 1
category_list = data_loading.cat_names




np.random.seed(seed)
torch.manual_seed(seed)

tab_model = models.TabularModelCustom(category_list, category_groups_sizes, len(data_loading.cont_names), [200, 100], ps=[0.2, 0.2],
                                      embed_p=0.,bn_final=final_bn)



image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=False,num_classes=image_output_size)
image_model.to("cuda:0")

# image_model.load_state_dict(
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/image_output-bceloss200-1.pt')))
# tab_model.load_state_dict(
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/tab_output-bceloss200-1.pt')))
model = models.CustomMixedModelSingleImage(image_model, tab_model,image_output_size,class_p=0.)
model = model.to("cuda:0")





loss_fn = torch.nn.BCEWithLogitsLoss().cuda()





now = datetime.datetime.now()
creation_time = now.strftime("%H:%M")
tensorboard_writer = SummaryWriter(os.path.expanduser('~/projetos/data/csgo_analyze/experiment/tensorboard/single_imageclass/') +
                                   experiment_name+"/"+now.strftime("%Y-%m-%d")+"/"+creation_time +"/"+ "bn_final-"+str(final_bn)+
                                   "-seed-"+str(seed))


class TensorboardClass():
    def __init__(self, writer):
        self.i = 0
        self.writer = writer


tensorboard_class = TensorboardClass(tensorboard_writer)
# torch.set_num_threads(8)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl_mixed),
                                                epochs=n_epochs)
for epoch in range(n_epochs):
    training.one_epoch(loss_fn, model, train_dl_mixed, valid_dl_mixed, optimizer, tensorboard_class,scheduler)
    torch.save(model.tab_model.state_dict(),
               str(pathlib.Path(path) / "tab_output-bceloss") + str(image_output_size)+"-" +str(epoch)+ ".pt")
    torch.save(model.image_model.state_dict(),
               str(pathlib.Path(path) / "image_output-bceloss") + str(image_output_size)+"-" +str(epoch)+  ".pt")







