import itertools
import logging
import random

import pandas as pd
import torch
import datetime
import pathlib
import os
import PIL
import torchvision
import numpy as np
import data_loading
import models
import training

path = "/home/marcel/projetos/data/csgo_analyze/processed/val"
image_files = data_loading.get_files(path, extensions=['.jpg'])
tabular_files = data_loading.get_files(path, extensions=['.csv'])
print(len(image_files))
print(len(tabular_files))
bs = 400
final_bn=True
image_output_size=200
seed = 42
experiment_name = "model_training"
device = "cuda:0"


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.manual_seed(seed)
cut_size = 0
n_epochs =3
lr=5e-4
img_sizes = [600,400,200]
seeds = [42,30,13]
logging.basicConfig(filename='single_image_eval.log', level=logging.INFO)
img_size = 200

if cut_size != 0:
    randomlist = random.sample(range(len(tabular_files)), cut_size)
    tabular_files = [tab_file for i,tab_file in enumerate(tabular_files) if i in randomlist]


print(image_files[0])
print(data_loading.fileLabeller(image_files[0]))
tabular_files = [fileName for fileName in tabular_files if pathlib.Path(fileName).name == "periodic_data.csv"]

full_csv=data_loading.filterTabularData(tabular_files,data_loading.columns)

filtered_image_files = data_loading.filterImageData(full_csv)


# filtered_image_files=filtered_image_files[:2000]
# full_csv=full_csv.iloc[:2000,:]
splits = data_loading.folderSplitter(filtered_image_files)
#splits = data_loading.roundSplitter(filtered_image_files)

assert filtered_image_files[203]==full_csv.iloc[203, -2]

# class groups:
# mainweapon, secweapon,flashbangs,hassmoke,hasmolotov,hashe,hashelmet,hasc4,hasdefusekit

distinct_matches = list(set(pathlib.Path(tab_file).parent.parent for tab_file in tabular_files))
distinct_rounds = list(set(pathlib.Path(tab_file).parent for tab_file in tabular_files))

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

valid_tabular = full_csv
valid_images = filtered_image_files

cat_transforms = TransformPipeline([Categorize(data_loading.x_category_map), ToTensor])
cont_transforms = TransformPipeline([Normalize(valid_tabular.iloc[:, :-2], category_map=data_loading.x_category_map), ToTensor])
label_transforms = TransformPipeline([Categorize(data_loading.y_category_map, multicat=False), ToTensor])
# image_transforms = torchvision.transforms.Compose(
#     [torchvision.transforms.Resize(200), torchvision.transforms.ToTensor(),
#      torchvision.transforms.Normalize((0.0019, 0.0040, 0.0061), (0.0043, 0.0084, 0.0124))])

#for seed,img_size in itertools.product(seeds,img_sizes):

image_transforms = [torchvision.transforms.ToTensor(),#data_loading.transposeLycon,
     torchvision.transforms.Normalize((255*0.0019, 255*0.0040, 255*0.0061), (255*0.0043, 255*0.0084, 255*0.0124))]
if img_size != 800:
    image_transforms = [data_loading.resize_lycon(img_size)]+image_transforms
image_transforms = torchvision.transforms.Compose(image_transforms)

valid_y_series = valid_tabular["winner"]
valid_x_tabular = valid_tabular.iloc[:, :-2]
valid_dataset = data_loading.CSGORoundsDatasetSingleImage(valid_images, valid_x_tabular, valid_y_series, image_transforms,
                                  cat_transforms, cont_transforms, label_transforms, data_loading.x_category_map, data_loading.y_category_map)


valid_dl_mixed = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False)           


group_count = 0
category_groups_sizes = {}
for class_group, class_group_categories in data_loading.x_category_map.items():
    category_groups_sizes[class_group] = (group_count, len(class_group_categories))
    group_count += 1
category_list = data_loading.cat_names




np.random.seed(seed)
torch.manual_seed(seed)

tab_model = models.TabularModelCustom(category_list, category_groups_sizes, len(data_loading.cont_names), [200, 100], ps=[0.2, 0.2],
                                      embed_p=0.2,bn_final=final_bn)



image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=False,num_classes=image_output_size)
image_model.to(device)


# tab_model.load_state_dict(
#     torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed_test/tab_output-bceloss200-1.pt')))
model = models.CustomMixedModelSingleImage(image_model, tab_model,image_output_size,class_p=0.2)
model = model.to(device)


model.load_state_dict(
    torch.load(os.path.expanduser('~/projetos/data/csgo_analyze/processed/full_model-bceloss200-0.pt')))


loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

def validation_cycle(image_files,valid_data_loader, model, loss_fn,device,**kwargs):
    model.eval()
    countImage=0
    for t, batch in enumerate(valid_data_loader):

        with torch.no_grad():
            loss,y_pred,y = training.generic_forwardpass(batch,model,loss_fn,device,**kwargs)
            data = {'File': image_files[countImage:(countImage+y_pred.shape[0])],
                    'Pred': torch.flatten(torch.sigmoid(y_pred)).cpu().detach().numpy()
                    }
            df = pd.DataFrame(data, columns = ['File', 'Pred'])
            if t % 5 == 0:
                logging.info("%s / %s = %s", t, len(valid_data_loader), loss.item())
            acc = training.accuracy(y_pred, y)
            countImage+=y_pred.shape[0]

validation_cycle(valid_images,valid_dl_mixed,model,loss_fn,device)

