import os
import mne.io as curry
import pickle

folder_path1  = "C:\\Users\\jbhan\\Desktop\\AA_CESA-2-DATA-EEG-Resting (Anden del)\\"
folder_path2 = "C:\\Users\\jbhan\\Desktop\\AA_CESA-2-DATA-EEG-Resting\\"

# Create lists of filenames in each folder. Only use files that starts with "S" and ends with ".dat"
file_names1 = [f for f in os.listdir(folder_path1) if f.endswith('.dat') and f.startswith('S')]
file_names2 = [f for f in os.listdir(folder_path2) if f.endswith('.dat') and f.startswith('S')]
# Delete the following fiels from their respective folders. They return errors...
if 'S26_12604_R001.dat' in file_names2:
    file_names2.remove('S26_12604_R001.dat')
if "S195_11851_R001.dat" in file_names1:
    file_names1.remove("S195_11851_R001.dat")
if "S309_11498_R001.dat" in file_names1:
    file_names1.remove("S309_11498_R001.dat")
if "S310_16627_R001.dat" in file_names1:
    file_names1.remove("S310_16627_R001.dat")
if "S311_11302_R001.dat" in file_names1:
    file_names1.remove("S311_11302_R001.dat")

# Get id that is written between underscores (starts with underscore and ends with underscore)
id_list1 = [f.split('_')[1].split('.')[0] for f in file_names1]
id_list2 = [f.split('_')[1].split('.')[0] for f in file_names2]

# Create a dictionary with id as key and file name as value
id_dict1 = dict(zip(id_list1, file_names1))
id_dict2 = dict(zip(id_list2, file_names2))
# Concatenate dict value to respective folder path
file_names = [folder_path1 + id_dict1[id] for id in id_list1] + [folder_path2 + id_dict2[id] for id in id_list2]
# Change all double backslashes to single forward slashes
file_names = [f.replace('\\','/') for f in file_names]
# Add to dictionary with id as key and file name as value
id_dict = dict(zip(id_list1 + id_list2, file_names))

valid_files = []
for key,value in id_dict.items():
    file_path = value
    raw_data = curry.read_raw_curry(file_path, preload=False, verbose=None)
    # if file includes 'HEO' 'VEO' channel name, then it is a valid file
    if 'HEO' in raw_data.ch_names and 'VEO' in raw_data.ch_names:
        valid_files.append(file_path)


# only keep the files in id_dict that are in valid_files
id_dict = {key:value for key,value in id_dict.items() if value in valid_files}
# pickle the id_dict
with open('data/valid_files.pkl', 'wb') as f:
    pickle.dump(id_dict, f)

# print keys of id_dict
print(len(id_dict.keys()))

