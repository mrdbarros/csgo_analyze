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
path = "/home/marcel/projetos/data/csgo_analyze/processed_test"
image_files = get_image_files(path)
tabular_files = get_files(path, extensions=['.csv'])
print(len(image_files))
print(len(tabular_files))
bs=32

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
for i,tab_file in enumerate(tabular_files):
    if i%50==0:
        print("processing file:",i,"of",len(tabular_files))
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
#class groups:
#mainweapon, secweapon,flashbangs,hassmoke,hasmolotov,hashe,hashelmet,hasc4,hasdefusekit

class_groups = {"mainweapon":L([101,102,103,104,105,106,107,
201,202,203,204,205,206,
301,302,303,304,305,306,307,308,309,310,311]),
"secweapon":L(0,1,2,3,4,5,6,7,8,9,10),
"flashbangs":L(0,1,2),
"hassmoke":L(0,1),
"hasmolotov":L(0,1),
"hashe":L(0,1),
"hashelmet":L(0,1),
"hasc4":L(0,1),
"hasdefusekit":L(0,1)}

# %%


def _apply_cats (voc, add, c):
    if not is_categorical_dtype(c):
        return pd.Categorical(c, categories=voc[c.name][add:]).codes+add
    return c.cat.codes+add #if is_categorical_dtype(c) else c.map(voc[c.name].o2i)
def _decode_cats(voc, c): return c.map(dict(enumerate(voc[c.name].items)))
labels=[1,2,3]
# Cell
class Categorify_Custom(TabularProc):
    "Transform the categorical variables to something similar to `pd.Categorical`"
    order = 1

    def setups(self, to,class_groups=class_groups):
        
        self.classes = {n:class_groups[n[n.rfind("_")+1:]] for n in to.cat_names}
        self.class_groups = class_groups
        

    def encodes(self, to): to.transform(to.cat_names, partial(_apply_cats, self.classes, 0))
    def decodes(self, to): to.transform(to.cat_names, partial(_decode_cats, self.classes))
    def __getitem__(self,k): return self.classes[k]


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
procs = [Categorify_Custom, Normalize]
dls_tabular = TabularDataLoaders.from_df(full_csv, path, procs=procs, cont_names=cont_names,cat_names=cat_names,
                                 y_names="winner", bs=bs,valid_idx=splits[1],device=torch.device('cuda:0'))


# %%

dsets = Datasets(filtered_image_files, [[PILImage.create], [fileLabeller, Categorize]],splits=splits)
item_tfms = [Resize(200),ToTensor]
batch_tfms = [IntToFloatTensor, Normalize.from_stats(*imagenet_stats)]
dls_image = dsets.dataloaders(after_item=item_tfms, after_batch=batch_tfms, bs=bs, num_workers=8,device=torch.device("cuda"))


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
tabular_classes = dls_tabular.train_ds.classes

# %%
#class groups:
#mainweapon, secweapon,flashbangs,hassmoke,hasmolotov,hashe,hashelmet,hasc4,hasdefusekit
tabular_classes

# %%
b=train_mixed_dl.one_batch()
b


# %%
train_mixed_dl.show_batch()

# %%
class TabularModelCustom(Module):
    "Basic model for tabular data."
    def __init__(self, classes,class_groups_sizes, n_cont, layers, ps=None, embed_p=0., 
    use_bn=True, bn_final=False, bn_cont=True):
        
        ps = ifnone(ps, [0]*len(layers))

        class_group_map = {}
        for i,cat in enumerate(classes):
            class_group = cat[cat.rfind("_")+1:]
            class_group_index,_ = class_groups_sizes[class_group]
            if class_group_index in class_group_map:
                class_group_map[class_group_index].append(i)
            else:
                class_group_map[class_group_index]=[i]
        self.class_group_map = class_group_map
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(index_ni[1], emb_sz_rule(index_ni[1])) for _,index_ni in class_groups_sizes.items() if index_ni[1]>3])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None

        binary_size = sum(len(class_group_map[i]) for i in range(len(self.embeds),len(class_group_map)))
        n_emb = sum(e.embedding_dim*len(class_group_map[i]) for i,e in enumerate(self.embeds))+binary_size
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)]
        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a)
                       for i,(p,a) in enumerate(zip(ps,actns))]
    
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        

        if self.n_emb != 0:
            x_cat_binary=[]
            for i in range(len(self.embeds),len(self.class_group_map)):
                x_cat_binary+=self.class_group_map[i]
            with torch.no_grad():
                x_cat_binary = x_cat[:,x_cat_binary].float()
            x_cat_nonbinary = [torch.flatten(e(x_cat[:,self.class_group_map[i]]),start_dim=1) for i,e in enumerate(self.embeds)]
            x = torch.cat(x_cat_nonbinary+[x_cat_binary], 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)
group_count=0
class_groups_sizes={}
for class_group,class_group_categories in class_groups.items():
    class_groups_sizes[class_group] = (group_count,len(class_group_categories))
    group_count+=1

tab_model = TabularModelCustom(tabular_classes,class_groups_sizes,len(cont_names),[200,100],ps=[0.2,0.2])
# %%
class CustomMixedModel(nn.Module):
    def __init__(self, image_model,tab_model):
        super(CustomMixedModel,self).__init__()
        self.image_model = image_model
        #embedding types are primaries, secondaries, flashbangs and binaries
        #self.classifier = TabularModel_NoCat(emb_sizes,1536, 30,[400],ps=[0.1],use_bn=False)
        self.tab_model = tab_model
        #n_emb = sum(e.embedding_dim for e in self.embeds)
        self.classifier = nn.Sequential(*[LinBnDrop(1200,2,act=None,p=0.),SigmoidRange(0.0,1.0)])


    def forward(self, input_cat,input_cont,input_image):
        
        
        output_tabular = self.tab_model(input_cat,input_cont)
        output_image =self.image_model(input_image)
        logits = self.classifier(torch.cat((output_tabular,output_image), dim=1))

        return logits


# %%
image_model =xresnet34()
image_model.to("cuda:0")
model = CustomMixedModel(image_model,tab_model)
model = model.to("cuda:0")


# %%
now= datetime.datetime.now()
creation_time = now.strftime("%H:%M")
writer = SummaryWriter( os.path.expanduser('~/projetos/data/csgo_analyze/experiment/tensorboard/')+
                        now.strftime("%Y-%m-%d-")+creation_time)
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
        if self.recorder.metric_names.index("accuracy")<len(self.recorder.log):
            self.tensorboard_writer.add_scalar(str(self.lr_sequence)+" loss: ",self.recorder.log[self.recorder.metric_names.index("valid_loss")],self.count)
            self.tensorboard_writer.add_scalar(str(self.lr_sequence)+" accuracy: ",self.recorder.log[self.recorder.metric_names.index("accuracy")],self.count)
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
learn.fit_one_cycle(10, lr_sequence[2])


# %%
learn.show_results()


# %%

learn.fit_one_cycle(3, 5e-5)


# %%
learn.cbs=learn.cbs[:-1]

