import data_loading
import pathlib
import shutil

path = "/home/marcel/projetos/data/csgo_analyze/processed"
train_path = path+"/train"
val_path = path+"/val"

image_files = data_loading.get_files(path, extensions=['.jpg'])
tabular_files = data_loading.get_files(path, extensions=['.csv'])
tabular_files = [fileName for fileName in tabular_files if pathlib.Path(fileName).name == "periodic_data.csv"]
print(len(image_files))
print(len(tabular_files))

distinct_matches = list(set(pathlib.Path(tab_file).parent.parent for tab_file in tabular_files))
train_matches,val_matches = data_loading.randomSplitter(distinct_matches, valid_pct=0.15)


for i,match in enumerate(distinct_matches):
    if i in train_matches:
        path_to_move = train_path
    else:
        path_to_move = val_path
    folder_number = int(match.parts[-1])
    while True:
        if not pathlib.Path(path_to_move + "/" + match.parts[-2] + "/" + str(folder_number)).exists():
            shutil.move(match, path_to_move + "/" + match.parts[-2] + "/" + str(folder_number))
            break
        else:
            folder_number+=1

    
print(distinct_matches)