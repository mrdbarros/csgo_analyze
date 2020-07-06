# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

%load_ext autoreload
%autoreload 2


# %%
# default_exp core

# %% [markdown]
# # module name here
# 
# > API details.

# %%
#hide
from nbdev.showdoc import *


# %%
from fastai2.tabular.all import *
from fastai2.vision.all import *
from fastai2.data.load import _FakeLoader, _loaders
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datetime


# %%
from fastai2.data.load import _FakeLoader, _loaders
class MixedDL():
    def __init__(self, tab_dl:TabDataLoader, vis_dl:TfmdDL, device='cuda:0'):
        "Stores away `tab_dl` and `vis_dl`, and overrides `shuffle_fn`"
        self.device = device
        tab_dl.shuffle_fn = self.shuffle_fn
        vis_dl.shuffle_fn = self.shuffle_fn
        self.dls = [tab_dl, vis_dl]
        self.count = 0
        self.fake_l = _FakeLoader(self, False, 0, 0)


# %%
@patch
def shuffle_fn(x:MixedDL, idxs):
        "Generates a new `rng` based upon which `DataLoader` is called"
        if x.count == 0: # if we haven't generated an rng yet
            x.rng = x.dls[0].rng.sample(idxs, len(idxs))
            x.count += 1
            return x.rng
        else:
            x.count = 0
            return x.rng

@patch
def __iter__(dl:MixedDL):
        "Iterate over your `DataLoader`"
        z = zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in dl.dls])
        for b in z:
            if dl.device is not None:
                b = to_device(b, dl.device)
            batch = []
            batch.extend(dl.dls[0].after_batch(b[0])[:2]) # tabular cat and cont
            batch.append(dl.dls[1].after_batch(b[1][0])) # Image
            batch.append(b[1][1]) # y
            yield tuple(batch)

@patch
def one_batch(x:MixedDL):
        "Grab a batch from the `DataLoader`"
        with x.fake_l.no_multiproc(): res = first(x)
        if hasattr(x, 'it'): delattr(x, 'it')
        return res

@patch
def __len__(x:MixedDL): return len(x.dls[0])

@patch
def to(x:MixedDL, device): x.device = device

@patch
def show_batch(x:MixedDL):
    "Show a batch from multiple `DataLoaders`"
    for dl in x.dls:
        dl.show_batch()

@patch
def show_results(x:MixedDL,b,out,**kwargs):
    "Show a batch from multiple `DataLoaders`"
    for i,dl in enumerate(x.dls):
        if i == 0:
            dl.show_results(b=b[:2]+(b[3],),out=out,**kwargs)
        else:
            dl.show_results(b=b[2:],out=out,**kwargs)

@patch
def new(x:MixedDL,*args,**kwargs):
    "Show a batch from multiple `DataLoaders`"
    new_dls = [dl.new(*args,**kwargs) for dl in x.dls]
    res=MixedDL(*new_dls)
    return res


# %%
path = "/home/marcel/projetos/data/csgo_analyze/processed_test/de_mirage"
image_files = get_image_files(path)
tabular_files = get_files(path, extensions=['.csv'])
print(len(image_files))
print(len(tabular_files))


# %%
def fileLabeller(o,**kwargs):
    winnerFile = Path(o).parent/"winner.txt"
    if os.path.isfile(winnerFile):
        f = open(winnerFile, "r")
        winner = f.readline()
        f.close()
    else:
        winner="na"
    return winner

print(image_files[0])
print(fileLabeller(image_files[0]))


# %%
columns = ["t_1","t_2","t_3","t_4","t_5","ct_1","ct_2","ct_3","ct_4","ct_5",
    "t_1_blindtime", "t_2_blindtime", "t_3_blindtime", "t_4_blindtime", "t_5_blindtime",
	"ct_1_blindtime", "ct_2_blindtime", "ct_3_blindtime", "ct_4_blindtime", "ct_5_blindtime",
    "t_1_mainweapon", "t_1_secweapon", "t_1_flashbangs", "t_1_hassmoke", "t_1_hasmolotov", "t_1_hashe", "t_1_armor", "t_1_hashelmet", "t_1_hasc4",
		"t_2_mainweapon", "t_2_secweapon", "t_2_flashbangs", "t_2_hassmoke", "t_2_hasmolotov", "t_2_hashe", "t_2_armor", "t_2_hashelmet", "t_2_hasc4",
		"t_3_mainweapon", "t_3_secweapon", "t_3_flashbangs", "t_3_hassmoke", "t_3_hasmolotov", "t_3_hashe", "t_3_armor", "t_3_hashelmet", "t_3_hasc4",
		"t_4_mainweapon", "t_4_secweapon", "t_4_flashbangs", "t_4_hassmoke", "t_4_hasmolotov", "t_4_hashe", "t_4_armor", "t_4_hashelmet", "t_4_hasc4",
		"t_5_mainweapon", "t_5_secweapon", "t_5_flashbangs", "t_5_hassmoke", "t_5_hasmolotov", "t_5_hashe", "t_5_armor", "t_5_hashelmet", "t_5_hasc4",
		"ct_1_mainweapon", "ct_1_secweapon", "ct_1_flashbangs", "ct_1_hassmoke", "ct_1_hasmolotov", "ct_1_hashe", "ct_1_armor", "ct_1_hashelmet", "ct_1_hasdefusekit",
		"ct_2_mainweapon", "ct_2_secweapon", "ct_2_flashbangs", "ct_2_hassmoke", "ct_2_hasmolotov", "ct_2_hashe", "ct_2_armor", "ct_2_hashelmet", "ct_2_hasdefusekit",
		"ct_3_mainweapon", "ct_3_secweapon", "ct_3_flashbangs", "ct_3_hassmoke", "ct_3_hasmolotov", "ct_3_hashe", "ct_3_armor", "ct_3_hashelmet", "ct_3_hasdefusekit",
		"ct_4_mainweapon", "ct_4_secweapon", "ct_4_flashbangs", "ct_4_hassmoke", "ct_4_hasmolotov", "ct_4_hashe", "ct_4_armor", "ct_4_hashelmet", "ct_4_hasdefusekit",
		"ct_5_mainweapon", "ct_5_secweapon", "ct_5_flashbangs", "ct_5_hassmoke", "ct_5_hasmolotov", "ct_5_hashe", "ct_5_armor", "ct_5_hashelmet", "ct_5_hasdefusekit",
        "round_time",
        'related_image','winner']
full_csv = pd.DataFrame(columns=columns)
for tab_file in tabular_files:
    if not os.stat(tab_file).st_size == 0 and os.path.isfile(tab_file.parent/"winner.txt"):
        new_csv = pd.read_csv(tab_file)
        new_csv['index']=new_csv.index
        new_csv['related_image'] = str(tab_file.parent)+"/output_map"+new_csv['index'].astype(str)+".jpg"
        new_csv['winner'] = fileLabeller(tab_file)
        new_csv=new_csv.drop(columns=["index"])
        new_csv.columns=columns
        full_csv=full_csv.append(new_csv)


# %%
filtered_image_files = L()
for image_file in image_files:
    if fileLabeller(image_file) in ["t","ct"] and not os.stat(image_file.parent/"tabular.csv").st_size == 0 and str(image_file) in full_csv['related_image'].values:
        filtered_image_files.append(image_file)


# %%
def roundSplitter(filtered_image_files):
    uniqueList = list(set([Path(o).parent for o in filtered_image_files]))
    splits=RandomSplitter()(uniqueList)
    train_image_files=L()
    valid_image_files=L()
    for i,o in enumerate(filtered_image_files):
        if uniqueList.index(Path(o).parent) in splits[0]:
            train_image_files+=i
        else:
            valid_image_files+=i
    return train_image_files,valid_image_files


# %%
filtered_image_files.sort()
full_csv=full_csv.sort_values(by=['related_image'])
#filtered_image_files=filtered_image_files[:2000]
#full_csv=full_csv.iloc[:2000,:]
splits=roundSplitter(filtered_image_files)


# %%
print(filtered_image_files[55])
full_csv.iloc[55,:]


# %%
cont_names = ['t_1', 't_2','t_3','t_4','t_5',
    'ct_1','ct_2','ct_3','ct_4','ct_5',
    "t_1_blindtime", "t_2_blindtime", "t_3_blindtime", "t_4_blindtime", "t_5_blindtime",
	"ct_1_blindtime", "ct_2_blindtime", "ct_3_blindtime", "ct_4_blindtime", "ct_5_blindtime",
    "t_1_armor","t_2_armor","t_3_armor","t_4_armor","t_5_armor",
    "ct_1_armor","ct_2_armor","ct_3_armor","ct_4_armor","ct_5_armor","round_time"]
cat_names = ["t_1_mainweapon", "t_1_secweapon", "t_1_flashbangs", "t_1_hassmoke", "t_1_hasmolotov", "t_1_hashe", "t_1_hashelmet", "t_1_hasc4",
		"t_2_mainweapon", "t_2_secweapon", "t_2_flashbangs", "t_2_hassmoke", "t_2_hasmolotov", "t_2_hashe", "t_2_hashelmet", "t_2_hasc4",
		"t_3_mainweapon", "t_3_secweapon", "t_3_flashbangs", "t_3_hassmoke", "t_3_hasmolotov", "t_3_hashe", "t_3_hashelmet", "t_3_hasc4",
		"t_4_mainweapon", "t_4_secweapon", "t_4_flashbangs", "t_4_hassmoke", "t_4_hasmolotov", "t_4_hashe", "t_4_hashelmet", "t_4_hasc4",
		"t_5_mainweapon", "t_5_secweapon", "t_5_flashbangs", "t_5_hassmoke", "t_5_hasmolotov", "t_5_hashe", "t_5_hashelmet", "t_5_hasc4",
		"ct_1_mainweapon", "ct_1_secweapon", "ct_1_flashbangs", "ct_1_hassmoke", "ct_1_hasmolotov", "ct_1_hashe", "ct_1_hashelmet", "ct_1_hasdefusekit",
		"ct_2_mainweapon", "ct_2_secweapon", "ct_2_flashbangs", "ct_2_hassmoke", "ct_2_hasmolotov", "ct_2_hashe", "ct_2_hashelmet", "ct_2_hasdefusekit",
		"ct_3_mainweapon", "ct_3_secweapon", "ct_3_flashbangs", "ct_3_hassmoke", "ct_3_hasmolotov", "ct_3_hashe", "ct_3_hashelmet", "ct_3_hasdefusekit",
		"ct_4_mainweapon", "ct_4_secweapon", "ct_4_flashbangs", "ct_4_hassmoke", "ct_4_hasmolotov", "ct_4_hashe", "ct_4_hashelmet", "ct_4_hasdefusekit",
		"ct_5_mainweapon", "ct_5_secweapon", "ct_5_flashbangs", "ct_5_hassmoke", "ct_5_hasmolotov", "ct_5_hashe", "ct_5_hashelmet", "ct_5_hasdefusekit"]
for cat in cont_names:
    full_csv[cat]=full_csv[cat].astype(np.float)
procs = [Categorify, Normalize]
dls_tabular = TabularDataLoaders.from_df(full_csv, path, procs=procs, cont_names=cont_names,cat_names=cat_names,
                                 y_names="winner", bs=8,valid_idx=splits[1],device=torch.device('cuda:0'))


# %%

dsets = Datasets(filtered_image_files, [[PILImage.create], [fileLabeller, Categorize]],splits=splits)
item_tfms = [Resize(200),ToTensor]
batch_tfms = [IntToFloatTensor, Normalize.from_stats(*imagenet_stats)]
dls_image = dsets.dataloaders(after_item=item_tfms, after_batch=batch_tfms, bs=8, num_workers=8,device=torch.device("cuda"))


# %%
train_mixed_dl = MixedDL(dls_tabular.train, dls_image.train)
train_mixed_dl.to(torch.device('cuda:0'))
valid_mixed_dl = MixedDL(dls_tabular.valid,dls_image.valid)
valid_mixed_dl.to(torch.device('cuda:0'))
dls = DataLoaders(train_mixed_dl, valid_mixed_dl,device=torch.device('cuda:0'))


# %%
dls_image.one_batch()


# %%
dls_tabular.one_batch()


# %%
b=train_mixed_dl.one_batch()
b


# %%
train_mixed_dl.show_batch()


# %%
class CustomMixedModel(nn.Module):
    def __init__(self, resNet):
        super(CustomMixedModel,self).__init__()
        self.resNet = resNet
        #self.classifier = TabularModel_NoCat(emb_sizes,1536, 30,[400],ps=[0.1],use_bn=False)
        emb_sizes=[(30,5),(11,3),(4,2)]
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_sizes])
        #n_emb = sum(e.embedding_dim for e in self.embeds)
        self.intermediate_linear = nn.Sequential(torch.nn.Linear(191,400),nn.ReLU())
        self.classifier=torch.nn.Linear(1400,2)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_cat,input_cont,input_image):
        main_weapon_fields = L([i*8 for i in range(10)])
        sec_weapon_fields = L([i*8+1 for i in range(10)])
        flash_fields = L([i*8+2] for i in range(10))
        other_fields = [i for i in range(80) if i not in main_weapon_fields+
            sec_weapon_fields+flash_fields]
        main_weapon_input = input_cat[:,main_weapon_fields]
        sec_weapon_input = input_cat[:,sec_weapon_fields]
        flash_input = input_cat[:,flash_fields]
        other_fields = input_cat[:,other_fields].float()
        main_weapon_input = torch.flatten(self.embeds[0](main_weapon_input), start_dim=1)
        
        sec_weapon_input = torch.flatten(self.embeds[1](sec_weapon_input), start_dim=1)
        flash_input = torch.flatten(self.embeds[2](flash_input), start_dim=1)
        output_image =self.resNet(input_image)
        output_tabular = self.intermediate_linear(torch.cat((main_weapon_input, sec_weapon_input,
            flash_input,other_fields,input_cont), dim=1))

        output=self.dropout(torch.cat((output_tabular, output_image), dim=1))

        logits = self.classifier(output)
        return logits


# %%
image_model =xresnet34()
image_model.to("cuda:0")
model = CustomMixedModel(image_model)
model = model.to("cuda:0")


# %%
now= datetime.datetime.now()
creation_time = now.strftime("%H:%M")
writer = SummaryWriter( os.path.expanduser('~/projetos/data/csgo_analyze/experiment/tensorboard/')+
                        now.strftime("%Y-%m-%d"))
class TensorboardCallback(Callback):
    def __init__(self,tensorboard_writer,creation_time,lr_sequence,with_input=False,
                 with_loss=True, save_preds=False, save_targs=False, concat_dim=0):
        store_attr(self, "with_input,with_loss,save_preds,save_targs,concat_dim")
        self.tensorboard_writer=tensorboard_writer
        self.count=0
        self.creation_time = creation_time
        self.lr_sequence=lr_sequence

    def begin_batch(self):
        if self.with_input: self.inputs.append((to_detach(self.xb)))

    def begin_validate(self):
        "Initialize containers"
        # self.preds,self.targets = [],[]
        # if self.with_input: self.inputs = []
        if self.with_loss:  
            self.losses = []
            self.accuracy=[]

    def after_batch(self):
        if not self.training:
            "Save predictions, targets and potentially losses"

            # preds,targs = to_detach(self.pred),to_detach(self.yb)
            # if self.save_preds is None: self.preds.append(preds)
            # else: (self.save_preds/str(self.iter)).save_array(preds)
            # if self.save_targs is None: self.targets.append(targs)
            # else: (self.save_targs/str(self.iter)).save_array(targs[0])
            if self.with_loss:
                self.accuracy.append(self.metrics[0].value)
                self.losses.append(to_detach(self.loss))
    def after_validate(self):
        "Concatenate all recorded tensors"
        # if self.with_input:     self.inputs  = detuplify(to_concat(self.inputs, dim=self.concat_dim))
        # if not self.save_preds: self.preds   = detuplify(to_concat(self.preds, dim=self.concat_dim))
        # if not self.save_targs: self.targets = detuplify(to_concat(self.targets, dim=self.concat_dim))

        self.tensorboard_writer.add_scalar(self.creation_time+" lr: "+str(self.lr_sequence)+" loss: ",self.recorder.log[self.recorder.metric_names.index("valid_loss")],self.count)
        self.tensorboard_writer.add_scalar(self.creation_time+" lr: "+str(self.lr_sequence)+" accuracy: ",self.recorder.log[self.recorder.metric_names.index("accuracy")],self.count)
        self.count+=1

    def all_tensors(self):
        res = [None if self.save_preds else self.preds, None if self.save_targs else self.targets]
        if self.with_input: res = [self.inputs] + res
        if self.with_loss:  res.append(self.losses)
        return res

lr_sequence = [5e-3,5e-4,5e-5]
tensorboardcb = TensorboardCallback(writer,creation_time,lr_sequence)


# %%
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy,cbs=[tensorboardcb])


# %%
for lr in lr_sequence:
    learn.fit_one_cycle(2, lr)


# %%
learn.show_results()


# %%

learn.fit_one_cycle(3, 5e-5)


# %%
learn.cbs=learn.cbs[:-1]

# %%


# %%


# %%


# %%
