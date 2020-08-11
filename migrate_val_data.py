import data_loading
import pathlib
import shutil

path = "/home/marcel/projetos/data/csgo_analyze/processed_test"
train_path = path+"/train"
val_path = path+"/val"

image_files = data_loading.get_files(path, extensions=['.jpg'])
tabular_files = data_loading.get_files(path, extensions=['.csv'])
print(len(image_files))
print(len(tabular_files))

distinct_matches = list(set(pathlib.Path(tab_file).parent.parent for tab_file in tabular_files))
train_matches,val_matches = data_loading.randomSplitter(distinct_matches)


for i,match in enumerate(distinct_matches):
    if i in train_matches:
        path_to_move = train_path
    else:
        path_to_move = val_path
    shutil.move(match, path_to_move + "/" + match.parts[-2] + "/" + match.parts[-1])
print(distinct_matches)