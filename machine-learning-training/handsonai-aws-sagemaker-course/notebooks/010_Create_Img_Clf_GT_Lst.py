#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pwd')


# In[1]:


#%load_ext nb_black


# In[3]:


import os
import json
import random
import pandas as pd
import shutil


# In[6]:


base_path="flower_photos_90/img_clf_lst/"
subdirs = ["train_imgs", "valid_imgs", "train_annots", "valid_annots"]
for subdir in subdirs:
    path = base_path + "/" + subdir
    if not os.path.exists(path):
        print(f"creating directory {path}")
        os.makedirs(path)
    else:
       print(f"{path} already exist")


# In[7]:


gt_df = pd.read_csv(
    "flower_photos_90/img_clf_lst/all_images_gt/clf_labels.csv", header=None
)
gt_df.columns = ["image", "class"]
gt_df.head()


# In[8]:


train_valid_split = 0.7
nimages = gt_df["image"].nunique()
ntrain = int(train_valid_split * nimages)
nvalid = nimages - ntrain
print(nimages, ntrain, nvalid)


# In[9]:


gt_df["class"] = gt_df["class"].apply(lambda x: x[1:-1])
gt_df.head()


# In[10]:


labels = ["daisy", "dandelion", "roses"]
labels_map = {k: v for v, k in enumerate(labels)}
with open("flower_photos_90/img_clf_lst/labels_map.json", "w") as fp:
    json.dump(labels_map, fp)
labels_map


# In[11]:


gt_df["class_id"] = gt_df["class"].map(labels_map)
gt_df.head()


# In[12]:


gt_df = gt_df.sample(frac=1).reset_index(drop=True)
gt_df.head()


# In[13]:


gt_df["index"] = gt_df.index + 1
gt_df.head()


# In[14]:


sel_cols = ["index", "class_id", "image"]
gt_df[sel_cols].head(ntrain).to_csv(
    "flower_photos_90/img_clf_lst/train_annots/train.lst",
    sep="\t",
    index=False,
    header=False,
)
get_ipython().system('head -n 5 flower_photos_90/img_clf_lst/train_annots/train.lst')


# In[15]:


gt_df[sel_cols].tail(nvalid).to_csv(
    "flower_photos_90/img_clf_lst/valid_annots/valid.lst",
    sep="\t",
    index=False,
    header=False,
)
get_ipython().system('head -n 5 flower_photos_90/img_clf_lst/valid_annots/valid.lst')


# In[16]:


train_df = pd.read_csv(
    "flower_photos_90/img_clf_lst/train_annots/train.lst", sep="\t", header=None
)
images = list(train_df[2].values)
for image in images:
    shutil.copy(
        "flower_photos_90/img_clf_lst/all_images_gt/" + image,
        "flower_photos_90/img_clf_lst/train_imgs/",
    )


# In[17]:


valid_df = pd.read_csv(
    "flower_photos_90/img_clf_lst/valid_annots/valid.lst", sep="\t", header=None
)
images = list(valid_df[2].values)
for image in images:
    shutil.copy(
        "flower_photos_90/img_clf_lst/all_images_gt/" + image,
        "flower_photos_90/img_clf_lst/valid_imgs/",
    )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:





# In[ ]:





# In[ ]:




