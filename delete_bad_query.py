import pandas as pd
import os
badpics_file = open("/home/zlm/dataset/vessel_reid/need_to_delete.txt","r")
badpics_lines = badpics_file.readlines()
path_df = pd.read_csv("/home/zlm/dataset/vessel_reid/query_path.csv")


path_list = path_df['filepath'].values.tolist()
# queryname_list = path_list.split('/')[-1]
queryname_list = [line.split('/')[-1][:-4] for line in path_list]#列表解析


for badpic in badpics_lines:
    badpic = badpic.rstrip()
    try:
        index = queryname_list.index(badpic)
        os.remove(path_list[index])
        print("delete " + path_list[index])
    except:
        print(badpic+" not in list")
