from __future__ import division
import os
import pathlib
import pandas as pd
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
import torchvision
import PIL
import time
import random
import types
import collections
import numpy as np
from random import shuffle
import logging
import lycon

# from nvidia.dali.pipeline import Pipeline
# import nvidia.dali.ops as ops
# import nvidia.dali.types as types
# from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator


columns = ["round_time","bomb_timeticking","t_1", "t_2", "t_3", "t_4", "t_5", "ct_1", "ct_2", "ct_3", "ct_4", "ct_5",
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
           'related_image', 'winner']

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

cont_names = ['t_1', 't_2', 't_3', 't_4', 't_5',
                  'ct_1', 'ct_2', 'ct_3', 'ct_4', 'ct_5',
                  "t_1_blindtime", "t_2_blindtime", "t_3_blindtime", "t_4_blindtime", "t_5_blindtime",
                  "ct_1_blindtime", "ct_2_blindtime", "ct_3_blindtime", "ct_4_blindtime", "ct_5_blindtime",
                  "t_1_armor", "t_2_armor", "t_3_armor", "t_4_armor", "t_5_armor",
                  "ct_1_armor", "ct_2_armor", "ct_3_armor", "ct_4_armor", "ct_5_armor", "round_time","bomb_timeticking"]
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

def _get_files(p, fs, extensions=None):
    p = pathlib.Path(p)
    res = [p / f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res


def get_files(path_folder, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path_folder = pathlib.Path(path_folder)
    extensions = set(extensions)
    extensions = {e.lower() for e in extensions}
    folders = folders if not folders is None else []
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(os.walk(path_folder, followlinks=followlinks)):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) != 0 and i == 0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path_folder) if o.is_file()]
        res = _get_files(path_folder, f, extensions)
    return res


def fileLabeller(o, **kwargs):
    winnerFile = pathlib.Path(o).parent / "winner.txt"
    if os.path.isfile(winnerFile):
        f = open(winnerFile, "r")
        winner = f.readline()
        f.close()
    else:
        winner = "na"
    return winner


def filterTabularData(tabular_files,columns):
    full_csv = []
    round_winners = {}
    for i, tab_file in enumerate(tabular_files):
        if i % 50 == 0:
            logging.info("processing file: %s of %s", i, len(tabular_files))
        if not os.stat(tab_file).st_size == 0 and os.path.isfile(tab_file.parent / "winner.txt"):
            new_csv = pd.read_csv(tab_file)
            new_csv['index'] = new_csv.index
            new_csv['related_image'] = str(tab_file.parent) + "/output_map" + new_csv['index'].astype(str).str.pad(width=2,
                                                                                        fillchar="0") + ".jpg"
            winner = fileLabeller(tab_file)
            if winner in ["t","ct"]:
                new_csv['winner'] = winner
                round_winners[tab_file] = winner
                new_csv = new_csv.drop(columns=["index"])
                new_csv.columns = columns
                full_csv.append(new_csv)
    full_csv = pd.concat(full_csv, ignore_index=True).sort_values(by=['related_image'])
    return full_csv

def filterImageData(full_csv):
    filtered_image_files = list(full_csv['related_image'].values)
    return filtered_image_files

def randomSplitter(elements, valid_pct=0.2, seed=None, **kwargs):
    "Create function that splits `items` between train/val with `valid_pct` randomly."
    if seed is not None: torch.manual_seed(seed)
    rand_idx = [int(i) for i in torch.randperm(len(elements))]
    cut = int(valid_pct * len(elements))
    return rand_idx[cut:], rand_idx[:cut]


def roundSplitter(filtered_image_files):
    uniqueList = list(set([pathlib.Path(o).parent for o in filtered_image_files]))
    splits = randomSplitter(uniqueList)
    train_image_files = []
    valid_image_files = []
    for i, o in enumerate(filtered_image_files):
        if uniqueList.index(pathlib.Path(o).parent) in splits[0]:
            train_image_files += [i]
        else:
            valid_image_files += [i]
    return train_image_files, valid_image_files

def folderSplitter(filtered_image_files):
    train_image_files = []
    valid_image_files = []
    for i, o in enumerate(filtered_image_files):
        if pathlib.Path(o).parts[-5] == "train":
            train_image_files += [i]
        elif pathlib.Path(o).parts[-5] == "val":
            valid_image_files += [i]
        else: print("invalid folder")

    return train_image_files, valid_image_files

def ToTensor(o):
    return torch.from_numpy(o)



class Categorize():
    def __init__(self, category_map: list = None, ordered_category_names = None,multicat=True):
        self.category_map = category_map
        self.multicat = multicat
        self.ordered_category_names=ordered_category_names

    def get_cat_code(self,value_series):
        group_dict = {key_cat: i for i, key_cat in
                      enumerate(self.category_map[value_series.name[value_series.name.rfind("_") + 1:]])}
        return value_series.replace(group_dict)

    def __call__(self, df_subset: pd.DataFrame):
        categories = []
        if self.multicat:
            new_df = df_subset.apply(self.get_cat_code)
            return new_df.values

        else:
            category = self.category_map["winner"].index(df_subset)
            categories.append(category)
            return np.array(categories)


    def decode(self,in_ndarray):
        ret_array = np.empty_like(in_ndarray)

        if self.multicat:
            for i in range(len(in_ndarray)):
                for j in range(len(in_ndarray[0])):
                    ret_array[i,j] = self.decode_element(in_ndarray[i,j],self.ordered_category_names[j])

        else:
            ret_array[0] = self.category_map["winner"][in_ndarray]

        return ret_array


    def decode_element(self,element,category):
        group = category[category.rfind("_") + 1:]
        return self.category_map[group][element]



class Normalize():
    def __init__(self, tabular_df: pd.DataFrame, category_map):
        self.means = [tabular_df[column].mean() for column in tabular_df.columns
                      if not column[column.rfind("_") + 1:] in category_map]
        self.std = [tabular_df[column].std() for column in tabular_df.columns
                    if not column[column.rfind("_") + 1:] in category_map]

    def __call__(self, o: pd.DataFrame):
        ret = o.copy()
        for i, column in enumerate(ret.columns):
            ret[column] = (o[column] - self.means[i]) / self.std[i]
        return ret.astype("float32").values

    def decode(self,o:pd.DataFrame):
        ret = o.copy()
        for i, column in enumerate(ret.columns):
            ret[column] = (o[column]*self.std[i] + self.means[i])

        return ret


class TransformPipeline():
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, o):
        res = o

        for transform in self.transforms:
            res = transform(res)

        return res
    def decode(self,o):
        res = o
        for i, transform in reversed(list(enumerate(self.transforms))):
            res = transform(res)
        return res


def calcNormWeights(imageList):
    means = [0., 0., 0.]
    std = [0., 0., 0.]
    resize = torchvision.transforms.Resize(200)
    to_tensor = torchvision.transforms.ToTensor()
    for j, imagePath in enumerate(imageList):
        image = to_tensor(resize(PIL.Image.open(imagePath)))
        if j % 50 == 0:
            print("processing image", j, "of", len(imageList))
        for i in range(3):
            means[i] += image[:, :, i].mean()
    for i in range(3):
        means[i] = means[i] / len(imageList)

    for j, imagePath in enumerate(imageList):
        if j % 50 == 0:
            print("processing image", j, "of", len(imageList))
        image = to_tensor(resize(PIL.Image.open(imagePath)))
        for i in range(3):
            std[i] += ((image[:, :, i] - means[i]) ** 2).mean()

    for i in range(3):
        std[i] = np.sqrt(std[i] / len(imageList))

    return means, std


# print(calcNormWeights(filtered_image_files))
# resizes 200, only mirage
# means:[tensor(0.0019), tensor(0.0040), tensor(0.0061)]
# std:[tensor(0.0043), tensor(0.0084), tensor(0.0124)]

class NormalizeImage():
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std

    def __call__(self,tensor):
        tensor = torch.div(tensor,255.)
        return torch.stack([(tensor[:,i,:,:]-self.mean[i])/self.std[i] for i in range(3)],dim=1)

    def decode(self,tensor):
        return torch.stack([(tensor[:, i, :, :]*self.std[i] + self.mean[i]) for i in range(3)], dim=1)




class CSGORoundsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list, round_paths: list, tabular_x_data: pd.DataFrame, round_winners,seq_size,
                 image_transform=None, cat_transform=None,
                 cont_transform=None, label_transform=None, x_category_map: dict = None, y_category_map: dict = None,batch_transform=None,
                 is_eval=False):

        self.image_paths = image_paths
        cat_columns = [column for column in tabular_x_data.columns if column[column.rfind("_") + 1:] in x_category_map]
        cont_columns = [column for column in tabular_x_data.columns if
                        not column[column.rfind("_") + 1:] in x_category_map]
        self.cat_data = tabular_x_data[cat_columns]
        self.cont_data = tabular_x_data[cont_columns]
        self.y = round_winners
        self.x_round_list = round_paths
        self.n_samples = len(self.x_round_list)
        self.image_transform = image_transform
        self.cat_transform = cat_transform
        self.cont_transform = cont_transform
        self.label_transform = label_transform
        self.x_category_map = x_category_map
        self.y_category_map = y_category_map
        self.batch_transform = batch_transform
        self.is_eval = is_eval
        self.seq_size = seq_size


        if not self.cat_transform is None:
            self.cat_data = self.cat_transform(self.cat_data)

        if not self.cont_transform is None:

            self.cont_data = self.cont_transform(self.cont_data)



    def __getitem__(self, index: int):
        round_path = self.x_round_list[index]
        indices = [i for i, path_match in enumerate(self.image_paths)
                   if self.image_paths[i][:self.image_paths[i].rfind("/")] == round_path]
        if not self.is_eval:
            inner_size = random.randint(2, len(indices))
            indices = indices[max((inner_size-self.seq_size),0):inner_size]
        else:
            print(round_path)


        images_lycon = self.open_round_images(indices)

        if not self.image_transform is None:
            tensor_image = [self.image_transform(image) for image in images_lycon]

            tensor_image = self.batch_transform(np.stack(tensor_image).transpose(0,3,1,2))


        if not self.label_transform is None:

            y = self.label_transform(self.y[round_path][0])

        return self.cat_data[indices], self.cont_data[indices], tensor_image, y


    def __len__(self):
        return self.n_samples

    def open_round_images(self, indices):
        image_paths = [path_match for i, path_match in enumerate(self.image_paths) if i in indices]
        images_lycon = [lycon.load(image_path) for image_path in image_paths]
        return images_lycon


def resize_lycon(size):
    def _inner(img):
        return lycon.resize(img, width=size, height=size, interpolation=lycon.Interpolation.LINEAR)
    return _inner

def get_rounds_and_winners(tabular_df: pd.DataFrame):
    rounds = []
    winners = {}

    new_df = tabular_df[['related_image','winner']].copy()
    new_df['related_image']=new_df['related_image'].apply(lambda o: o[:o.rfind("/")])
    new_df=new_df.drop_duplicates()
    winners = new_df.set_index('related_image').T.to_dict("list")
    # for index, row in tabular_df.iterrows():
    #     round_path = row['related_image'][:row['related_image'].rfind("/")]
    #     rounds.append(round_path)
    #     winners[round_path] = row['winner']
    return list(new_df['related_image']), winners



def snapshot_sequence_batching(length,is_eval=False):
    def _inner(batch):  # cat,cont,image,y
        full_batch = list(zip(*batch))

        if is_eval:
            for i in range(len(full_batch)-1):
                full_batch[i]=[full_batch[i][0][:j] for j in range(1,full_batch[i][0].shape[0])]+[full_batch[i][0]]
            full_batch[-1]=[full_batch[-1][0] for _ in range(1,full_batch[0][-1].shape[0])]+[full_batch[-1][0]]


        cat_batch = []
        cont_batch = []
        attention_mask = []
        image_batch = []
        for i, image_sequence in enumerate(full_batch[2]):
            if length - image_sequence.shape[0] > 0:
                # attention_mask.append(torch.ones(length).bool())
                attention_mask.append(torch.zeros(length))
                attention_mask[i][:image_sequence.shape[0]] = 1
                cat_batch.append(full_batch[0][i])
                cont_batch.append(full_batch[1][i])
                image_batch.append(full_batch[2][i])
            else:
                # attention_mask.append(torch.zeros(length).bool())
                attention_mask.append(torch.ones(length))
                cat_batch.append(full_batch[0][i][:length])
                cont_batch.append(full_batch[1][i][:length])
                image_batch.append(full_batch[2][i][:length])

        cat_batch = torch.cat(cat_batch,dim=0)
        cont_batch = torch.cat(cont_batch, dim=0)
        image_batch = torch.cat(image_batch, dim=0)

        y_batch = default_collate(full_batch[3])
        attention_mask=default_collate(attention_mask)
        return cat_batch, cont_batch, image_batch, attention_mask, y_batch.float()

    return _inner

def prepare_and_pad_sequence(length):
    def _inner(input_embed,valid_sizes):
        sequence_batch = []
        for i in range(valid_sizes.shape[0]):
            interval_begin = torch.sum(valid_sizes[:i]).item() if i > 0 else 0
            sequence_element = input_embed[interval_begin:(interval_begin+valid_sizes[i])]
            if length - valid_sizes[i] > 0:
                pad_cat_cont = (0, 0, 0, length - valid_sizes[i])

                sequence_batch.append(torch.nn.functional.pad(sequence_element, pad_cat_cont, "constant", 0))  # cat

            else:
                sequence_batch.append(sequence_element)

        sequence_batch = torch.stack(sequence_batch,dim=0)

        return sequence_batch
    return _inner


def pad_snapshot_sequence(length,is_eval=False):
    def _inner(batch):  # cat,cont,image,y
        full_batch = list(zip(*batch))

        if is_eval:
            for i in range(len(full_batch)-1):
                full_batch[i]=[full_batch[i][0][:j] for j in range(1,full_batch[i][0].shape[0])]+[full_batch[i][0]]
            full_batch[-1]=[full_batch[-1][0] for _ in range(1,full_batch[0][-1].shape[0])]+[full_batch[-1][0]]


        cat_batch = []
        cont_batch = []
        attention_mask = []
        image_batch = []
        for i, image_sequence in enumerate(full_batch[2]):
            if length - image_sequence.shape[0] > 0:
                pad_cat_cont = (0, 0, 0, length - image_sequence.shape[0])
                pad_image = (0, 0, 0, 0, 0, 0, 0, length - image_sequence.shape[0])
                # attention_mask.append(torch.ones(length).bool())
                attention_mask.append(torch.zeros(length))
                attention_mask[i][:image_sequence.shape[0]] = 1
                cat_batch.append(torch.nn.functional.pad(full_batch[0][i], pad_cat_cont, "constant", 0))  # cat
                cont_batch.append(torch.nn.functional.pad(full_batch[1][i], pad_cat_cont, "constant", 0))  # cont
                image_batch.append(torch.nn.functional.pad(image_sequence, pad_image, "constant", 0))  # image
            else:
                # attention_mask.append(torch.zeros(length).bool())
                attention_mask.append(torch.ones(length))
                cat_batch.append(full_batch[0][i][:length])
                cont_batch.append(full_batch[1][i][:length])
                image_batch.append(full_batch[2][i][:length])

        cat_batch = default_collate(cat_batch)
        cont_batch = default_collate(cont_batch)
        y_batch = default_collate(full_batch[3])
        image_batch = default_collate(image_batch)
        attention_mask=default_collate(attention_mask)
        return cat_batch, cont_batch, image_batch, attention_mask, y_batch.float()

    return _inner

def transposeLycon(o):
    return o.transpose(0,3,1,2)


class CSGORoundsDatasetSingleImage(torch.utils.data.Dataset):
    def __init__(self, image_paths: list, tabular_x_data: pd.DataFrame, y_array, image_transform=None,
                 cat_transform=None,
                 cont_transform=None, label_transform=None, x_category_map: dict = None, y_category_map: dict = None):
        self.image_paths = image_paths
        cat_columns = [column for column in tabular_x_data.columns if column[column.rfind("_") + 1:] in x_category_map]
        cont_columns = [column for column in tabular_x_data.columns if
                        not column[column.rfind("_") + 1:] in x_category_map]
        self.cat_data = tabular_x_data[cat_columns]
        self.cont_data = tabular_x_data[cont_columns]
        self.y = y_array
        self.n_samples = len(self.image_paths)
        self.image_transform = image_transform
        self.cat_transform = cat_transform
        self.cont_transform = cont_transform
        self.label_transform = label_transform
        self.x_category_map = x_category_map
        self.y_category_map = y_category_map

    def __getitem__(self, index: int):
        image=lycon.load(self.image_paths[index])
        if not self.image_transform is None:
            tensor_image = self.image_transform(image)
        if not self.cat_transform is None:
            cat_data = self.cat_transform(self.cat_data.iloc[index, :])

        if not self.cont_transform is None:
            cont_data = self.cont_transform(self.cont_data.iloc[index, :])

        if not self.label_transform is None:
            y = self.label_transform(self.y.iloc[index])

        return cat_data, cont_data, tensor_image, y.float()

    def __len__(self):
        return self.n_samples


# class ExternalInputIterator(object):
#     def __init__(self, image_paths: list, round_paths: list, tabular_x_data: pd.DataFrame, round_winners,batch_size, cat_transform=None,
#                  cont_transform=None, label_transform=None, x_category_map: dict = None, y_category_map: dict = None,
#                  is_eval=False):
#
#
#
#         self.image_paths = image_paths
#         self.x_round_list = round_paths
#         self.n = len(self.x_round_list)
#         self.batch_size = batch_size
#         self.tabular_dataset = CSGORoundsDatasetNoImages(image_paths,round_paths,tabular_x_data,round_winners,
#                                                      cat_transform,cont_transform,label_transform,x_category_map,
#                                                      y_category_map,is_eval)
#
#     def __iter__(self):
#         self.i = 0
#         self.order = list(range(self.n))
#         random.shuffle(self.order)
#         return self
#
#     def __next__(self):
#         image_batch = []
#         cat_batch = []
#         cont_batch = []
#         y_batch = []
#         inner_size_batch = []
#         if self.i >= self.n:
#             raise StopIteration
#
#
#
#         for _ in range(self.batch_size):
#             round_path = self.x_round_list[self.order[self.i]]
#             indices = [i for i, path_match in enumerate(self.image_paths)
#                        if pathlib.Path(path_match).parent == round_path]
#             cat_data, cont_data, y,inner_size = self.tabular_dataset[self.order[self.i]]
#             round_images = []
#             for index in indices:
#                 f = open(self.image_paths[index], 'rb')
#                 round_images.append(np.frombuffer(f.read(), dtype = np.uint8))
#             image_batch.append(round_images)
#             cat_batch.append(cat_data)
#             cont_batch.append(cont_batch)
#             y_batch.append(y)
#             inner_size_batch.append(inner_size)
#             self.i = self.i + 1
#
#         return cat_batch,cont_batch,image_batch,y_batch,inner_size_batch
#
#     @property
#     def size(self,):
#         return self.n
#
#     next = __next__
#
# class ExternalSourcePipeline(Pipeline):
#     def __init__(self, batch_size,image_size, num_threads, device_id, external_data,seed):
#         super(ExternalSourcePipeline, self).__init__(batch_size,
#                                       num_threads,
#                                       device_id,
#                                       seed=seed)
#         self.batch_image = ops.ExternalSource()
#         self.batch_cat = ops.ExternalSource()
#         self.batch_cont = ops.ExternalSource()
#         self.batch_y = ops.ExternalSource()
#         self.batch_inner_size = ops.ExternalSource()
#
#         self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
#         self.res = ops.Resize(device="gpu", resize_shorter = image_size)
#
#         self.cast_float = ops.Cast(device="gpu",
#                              dtype=types.FLOAT)
#         self.cast_int = ops.Cast(device="gpu",
#                                    dtype=types.UINT8)
#         self.external_data = external_data
#         self.iterator = iter(self.external_data)
#
#     def define_graph(self):
#         self.images = self.batch_image()
#         self.categories = self.batch_cat()
#         self.cont = self.batch_cont()
#         self.ys = self.batch_y()
#         self.inner_sizes = self.batch_inner_size()
#         images = self.decode(self.images)
#         images = self.res(images)
#         images = self.cast_int(images)
#
#         categories = self.cast_int(self.categories)
#         cont = self.cast_float(self.cont)
#         ys = self.cast_int(self.ys)
#         return categories,cont,images,ys,self.inner_sizes
#
#     def iter_setup(self):
#         try:
#             cat_batch,cont_batch,image_batch,y_batch,inner_size_batch = self.iterator.next()
#             self.feed_input(self.images, image_batch)
#             self.feed_input(self.categories, cat_batch)
#             self.feed_input(self.cont, cont_batch)
#             self.feed_input(self.ys, y_batch)
#             self.feed_input(self.inner_sizes, inner_size_batch)
#
#         except StopIteration:
#             self.iterator = iter(self.external_data)
#             raise StopIteration

class CSGORoundsDatasetNoImages(torch.utils.data.Dataset):
    def __init__(self, image_paths: list, round_paths: list, tabular_x_data: pd.DataFrame, round_winners,
                 cat_transform=None,
                 cont_transform=None, label_transform=None, x_category_map: dict = None, y_category_map: dict = None,
                 is_eval=False):

        self.image_paths = image_paths
        cat_columns = [column for column in tabular_x_data.columns if column[column.rfind("_") + 1:] in x_category_map]
        cont_columns = [column for column in tabular_x_data.columns if
                        not column[column.rfind("_") + 1:] in x_category_map]
        self.cat_data = tabular_x_data[cat_columns]
        self.cont_data = tabular_x_data[cont_columns]
        self.y = round_winners
        self.x_round_list = round_paths
        self.n_samples = len(self.x_round_list)

        self.cat_transform = cat_transform
        self.cont_transform = cont_transform
        self.label_transform = label_transform
        self.x_category_map = x_category_map
        self.y_category_map = y_category_map

        self.is_eval = is_eval

    def __getitem__(self, index: int):
        round_path = self.x_round_list[index]
        indices = [i for i, path_match in enumerate(self.image_paths)
                   if pathlib.Path(path_match).parent == round_path]
        if not self.is_eval:
            inner_size = random.randint(2, len(indices))
            indices = indices[:inner_size]
        else:
            print(round_path)


        if not self.cat_transform is None:
            cat_data = self.cat_transform(self.cat_data.iloc[indices, :])

        if not self.cont_transform is None:

            cont_data = self.cont_transform(self.cont_data.iloc[indices, :])

        if not self.label_transform is None:

            y = self.label_transform(self.y[round_path])


        return cat_data, cont_data, y,inner_size