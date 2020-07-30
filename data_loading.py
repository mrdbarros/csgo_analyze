
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
            print("processing file:", i, "of", len(tabular_files))
        if not os.stat(tab_file).st_size == 0 and os.path.isfile(tab_file.parent / "winner.txt"):
            new_csv = pd.read_csv(tab_file)
            new_csv['index'] = new_csv.index
            new_csv['related_image'] = str(tab_file.parent) + "/output_map" + new_csv['index'].astype(str).str.pad(width=2,
                                                                                                                   fillchar="0") + ".jpg"
            winner = fileLabeller(tab_file)
            new_csv['winner'] = winner
            round_winners[tab_file] = winner
            new_csv = new_csv.drop(columns=["index"])
            new_csv.columns = columns
            full_csv.append(new_csv)
    full_csv = pd.concat(full_csv, ignore_index=True).sort_values(by=['related_image'])
    return full_csv

def filterImageData(image_files,full_csv):
    filtered_image_files = []
    for image_file in image_files:
        if fileLabeller(image_file) in ["t", "ct"] and not os.stat(image_file.parent / "tabular.csv").st_size == 0 and \
                str(image_file) in full_csv['related_image'].values:
            filtered_image_files.append(image_file)
        #else: print(image_file)
    filtered_image_files.sort()
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

def ToTensor(o):
    return torch.from_numpy(o)



class Categorize():
    def __init__(self, category_map: list = None, ordered_category_names = None,multicat=True):
        self.category_map = category_map
        self.multicat = multicat
        self.ordered_category_names=ordered_category_names



    def __call__(self, df_subset: pd.DataFrame):
        categories = []
        if self.multicat:
            for i, df_row in df_subset.iterrows():
                row_categories = []
                for cat in df_subset.columns:
                    group = cat[cat.rfind("_") + 1:]
                    category = self.category_map[group].index(df_row[cat])
                    row_categories.append(category)
                categories.append(row_categories)
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

        return torch.stack([(tensor[:,i,:,:]-self.mean[i])/self.std[i] for i in range(3)],dim=1)

    def decode(self,tensor):
        return torch.stack([(tensor[:, i, :, :]*self.std[i] + self.mean[i]) for i in range(3)], dim=1)



class CSGORoundsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list, round_paths: list, tabular_x_data: pd.DataFrame, round_winners,
                 image_transform=None, cat_transform=None,
                 cont_transform=None, label_transform=None, x_category_map: dict = None, y_category_map: dict = None,batch_transform=None):

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

    def __getitem__(self, index: int):
        round_path = self.x_round_list[index]
        indices = [i for i, path_match in enumerate(self.image_paths)
                   if pathlib.Path(path_match).parent == round_path]

        inner_size = random.randint(2, len(indices))
        indices = indices[:inner_size]
        transform_times = []
        transform_times.append(time.time())
        images = self.open_round_images(indices)
        transform_times.append(time.time())
        if not self.image_transform is None:
            tensor_image = torch.stack([self.image_transform(image) for image in images], dim=0)
            transform_times.append(time.time())
            tensor_image = self.batch_transform(tensor_image)
            transform_times.append(time.time())
        if not self.cat_transform is None:
            cat_data = self.cat_transform(self.cat_data.iloc[indices, :])

        if not self.cont_transform is None:

            cont_data = self.cont_transform(self.cont_data.iloc[indices, :])

        if not self.label_transform is None:

            y = self.label_transform(self.y[round_path])

        #print(np.diff(np.array(transform_times)))
        return cat_data, cont_data, tensor_image, y,inner_size

    def __len__(self):
        return self.n_samples

    def open_round_images(self, indices):
        image_paths = [path_match for i, path_match in enumerate(self.image_paths) if i in indices]
        images = [PIL.Image.open(image_path) for image_path in image_paths]
        return images


def get_rounds_and_winners(tabular_df: pd.DataFrame):
    rounds = []
    winners = {}
    for index, row in tabular_df.iterrows():
        round_path = pathlib.Path(row['related_image']).parent
        rounds.append(round_path)
        winners[round_path] = row['winner']
    return list(set(rounds)), winners

def pad_snapshot_sequence(length):
    def _inner(batch):  # cat,cont,image,y
        full_batch = list(zip(*batch))
        cat_batch = []
        cont_batch = []
        attention_mask = []
        image_batch = []
        for i, image_sequence in enumerate(full_batch[2]):
            if length - image_sequence.shape[0] > 0:
                pad_cat_cont = (0, 0, 0, length - image_sequence.shape[0])
                pad_image = (0, 0, 0, 0, 0, 0, 0, length - image_sequence.shape[0])
                attention_mask.append(torch.ones(length).bool())
                attention_mask[i][:image_sequence.shape[0]] = False
                cat_batch.append(torch.nn.functional.pad(full_batch[0][i], pad_cat_cont, "constant", 0))  # cat
                cont_batch.append(torch.nn.functional.pad(full_batch[1][i], pad_cat_cont, "constant", 0))  # cont
                image_batch.append(torch.nn.functional.pad(image_sequence, pad_image, "constant", 0))  # image
            else:
                attention_mask.append(torch.zeros(length).bool())
                cat_batch.append(full_batch[0][i][:length])
                cont_batch.append(full_batch[1][i][:length])
                image_batch.append(full_batch[2][i][:length])

        cat_batch = default_collate(cat_batch)
        cont_batch = default_collate(cont_batch)
        y_batch = default_collate(full_batch[3])
        image_batch = default_collate(image_batch)
        attention_mask=default_collate(attention_mask)
        return cat_batch, cont_batch, image_batch, attention_mask, y_batch

    return _inner