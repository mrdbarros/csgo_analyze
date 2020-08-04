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
bs = 60




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


full_csv=data_loading.filterTabularData(tabular_files,columns)

filtered_image_files = data_loading.filterImageData(image_files,full_csv)




filtered_image_files.sort()
full_csv = full_csv.sort_values(by=['related_image'])
# filtered_image_files=filtered_image_files[:2000]
# full_csv=full_csv.iloc[:2000,:]
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

class CSGORoundsDataset(torch.utils.data.Dataset):
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
        image = PIL.Image.open(self.image_paths[index])
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


train_tabular = full_csv.iloc[splits[0], :]
train_images = [image for i, image in enumerate(filtered_image_files) if i in splits[0]]
valid_tabular = full_csv.iloc[splits[1], :]
valid_images = [image for i, image in enumerate(filtered_image_files) if i in splits[1]]

cat_transforms = TransformPipeline([Categorize(x_category_map), ToTensor])
cont_transforms = TransformPipeline([Normalize(train_tabular.iloc[:, :-2], category_map=x_category_map), ToTensor])
label_transforms = TransformPipeline([Categorize(y_category_map, multicat=False), ToTensor])
image_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(200), torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.0019, 0.0040, 0.0061), (0.0043, 0.0084, 0.0124))])
train_y_series = train_tabular["winner"]
train_x_tabular = train_tabular.iloc[:, :-2]
train_dataset = CSGORoundsDataset(train_images, train_x_tabular, train_y_series, image_transforms,
                                  cat_transforms, cont_transforms, label_transforms, x_category_map, y_category_map)

valid_y_series = valid_tabular["winner"]
valid_x_tabular = valid_tabular.iloc[:, :-2]
valid_dataset = CSGORoundsDataset(valid_images, valid_x_tabular, valid_y_series, image_transforms,
                                  cat_transforms, cont_transforms, label_transforms, x_category_map, y_category_map)

print(train_dataset[2])
train_dl_mixed = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6)
valid_dl_mixed = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=6)

for image_output_size in range(200,1000,100):
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





    image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=False,num_classes=image_output_size)
    image_model.to("cuda:0")
    model = models.CustomMixedModelSingleImage(image_model, tab_model,image_output_size)
    model = model.to("cuda:0")








    def one_epoch(loss_fn, model, train_data_loader, valid_data_loader, optimizer, tensorboard_class):
        model.train()
        valid_loss = training.AverageMeter('Loss', ':.4e')
        valid_accuracy = training.AverageMeter('Acc', ':6.2f')
        for t, batch in enumerate(train_data_loader):

            x_input = batch[:-1]
            y = batch[-1]

            y = y.cuda()
            for i, x_tensor in enumerate(x_input):
                x_input[i] = x_tensor.cuda()
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(*x_input)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            if t % 30 == 0:
                print(t, "/", len(train_data_loader), loss.item())

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            # scheduler.step()

        model.eval()
        for t, batch in enumerate(valid_data_loader):

            x_input = batch[:-1]
            y = batch[-1]
            # Forward pass: compute predicted y by passing x to the model.
            y = y.cuda()
            for i, x_tensor in enumerate(x_input):
                x_input[i] = x_tensor.cuda()
            with torch.no_grad():
                y_pred = model(*x_input)

                # Compute and print loss.
                loss = loss_fn(y_pred, y)
                if t % 30 == 0:
                    print(t, "/", len(valid_data_loader), loss.item())
                acc = training.accuracy(y_pred, y)
                valid_accuracy.update(acc.item(), y.shape[0])
                valid_loss.update(loss.item(), y.shape[0])

        print(' * Acc {valid_accuracy.avg:.3f} Loss {valid_loss.avg:.3f}'
              .format(valid_accuracy=valid_accuracy, valid_loss=valid_loss))
        tensorboard_class.writer.add_scalar("loss:",
                                            valid_loss.avg, tensorboard_class.i)
        tensorboard_class.writer.add_scalar("accuracy:",
                                            valid_accuracy.avg, tensorboard_class.i)
        tensorboard_class.i += 1


    loss_fn = torch.nn.BCEWithLogitsLoss().cuda()



    lr=5e-4
    n_epochs = 3

    now = datetime.datetime.now()
    creation_time = now.strftime("%H:%M")
    tensorboard_writer = SummaryWriter(os.path.expanduser('~/projetos/data/csgo_analyze/experiment/tensorboard/') +
                                       now.strftime("%Y-%m-%d-") + "image_out-"+str(image_output_size))


    class TensorboardClass():
        def __init__(self, writer):
            self.i = 0
            self.writer = writer


    tensorboard_class = TensorboardClass(tensorboard_writer)
    # torch.set_num_threads(8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        one_epoch(loss_fn, model, train_dl_mixed, valid_dl_mixed, optimizer, tensorboard_class)
        torch.save(model.tab_model.state_dict(),
                   str(pathlib.Path(path) / "tab_output-bceloss") + str(image_output_size)+"-" +str(epoch)+ ".pt")
        torch.save(model.image_model.state_dict(),
                   str(pathlib.Path(path) / "image_output-bceloss") + str(image_output_size)+"-" +str(epoch)+  ".pt")







