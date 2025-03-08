#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# In[1]:


import os
import json
import random
import pandas as pd
import shutil


# In[2]:


base_path="flower_photos_90/img_clf_lst_from_subfolds/"
subdirs = ["train_imgs", "valid_imgs", "train_annots", "valid_annots"]
for subdir in subdirs:
    path = base_path + "/" + subdir
    if not os.path.exists(path):
        print(f"creating directory {path}")
        os.makedirs(path)
    else:
       print(f"{path} already exist")


# In[8]:


images = []
classes = []

for path, subdirs, files in os.walk(
    "flower_photos_90/img_clf_lst_from_subfolds/all_images"
):
    for file in files:
        # print(os.path.join(path, name))
        if file == ".DS_Store":
            continue
        else:
            images.append(file)
            #classes.append(path.split("/")[-1])
            # changed to '\' to deal with Windows
            classes.append(path.split("\\")[-1])

print(path)
print(images[:5])
print(classes[:5])


# In[9]:


gt_df = pd.DataFrame({"image": images, "class": classes})
gt_df.head()


# In[6]:


# gt_df = pd.read_csv(
#     "prepare_date/img_clf_lst/all_images_gt/clf_labels.csv", header=None
# )
# gt_df.columns = ["image", "class"]
# gt_df.head()


# In[10]:


train_valid_split = 0.7
nimages = gt_df["image"].nunique()
ntrain = int(train_valid_split * nimages)
nvalid = nimages - ntrain
print(nimages, ntrain, nvalid)


# In[8]:


# gt_df["class"] = gt_df["class"].apply(lambda x: x[1:-1])
# gt_df.head()


# In[12]:


labels = ["daisy", "dandelion", "roses"]
labels_map = {k: v for v, k in enumerate(labels)}
with open("flower_photos_90/img_clf_lst_from_subfolds/labels_map.json", "w") as fp:
    json.dump(labels_map, fp)
labels_map


# In[13]:


gt_df["class_id"] = gt_df["class"].map(labels_map)
gt_df.head()


# In[14]:


gt_df = gt_df.sample(frac=1).reset_index(drop=True)
gt_df.head()


# In[15]:


gt_df["index"] = gt_df.index + 1
gt_df.head()


# In[17]:


sel_cols = ["index", "class_id", "image"]
gt_df[sel_cols].head(ntrain).to_csv(
    "flower_photos_90/img_clf_lst_from_subfolds/train_annots/train.lst",
    sep="\t",
    index=False,
    header=False,
)
get_ipython().system('head -n 5 flower_photos_90/img_clf_lst/train_annots/train.lst')


# In[18]:


gt_df[sel_cols].tail(nvalid).to_csv(
    "flower_photos_90/img_clf_lst_from_subfolds/valid_annots/valid.lst",
    sep="\t",
    index=False,
    header=False,
)
get_ipython().system('head -n 5 flower_photos_90/img_clf_lst/valid_annots/valid.lst')


# In[20]:


tmp_path="flower_photos_90/img_clf_lst_from_subfolds/all_images_tmp"
if not os.path.exists(tmp_path):
    print(f"creating directory {tmp_path}")
    os.makedirs(tmp_path)
else:
    print(f"{tmp_path} already exist")


# In[21]:


#!mkdir flower_photos_90/img_clf_lst_from_subfolds/all_images_tmp
for label in labels:
    get_ipython().system('cp flower_photos_90/img_clf_lst_from_subfolds/all_images/{label}/* flower_photos_90/img_clf_lst_from_subfolds/all_images_tmp/')


# In[22]:


train_df = pd.read_csv(
    "flower_photos_90/img_clf_lst_from_subfolds/train_annots/train.lst",
    sep="\t",
    header=None,
)
images = list(train_df[2].values)
for image in images:
    shutil.copy(
        "flower_photos_90/img_clf_lst_from_subfolds/all_images_tmp/" + image,
        "flower_photos_90/img_clf_lst_from_subfolds/train_imgs/",
    )


# In[23]:


valid_df = pd.read_csv(
    "flower_photos_90/img_clf_lst/valid_annots/valid.lst", sep="\t", header=None
)
images = list(valid_df[2].values)
for image in images:
    shutil.copy(
        "flower_photos_90/img_clf_lst_from_subfolds/all_images_tmp/" + image,
        "flower_photos_90/img_clf_lst_from_subfolds/valid_imgs/",
    )


# In[24]:


get_ipython().system('rm -r flower_photos_90/img_clf_lst_from_subfolds/all_images_tmp')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:





# In[20]:




