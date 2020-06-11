#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def split_data(source,path,label_dir,split_size):
    
    dataset = []
    dir1=['Train','Test']
    dir2=[]
    
    for file in os.listdir(source):
        img = os.path.join(source,file)
        if(os.path.getsize(img) > 0):
            dataset.append(file)
        else:
            print("Image {} is corrupt".format(file))
            
    
    train_len = int(len(dataset) * split_size)
    test_len = int(len(dataset) - train_len)
    
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = shuffled_set[0:train_len]
    test_set = shuffled_set[-test_len:]
    
    #Makes Parent Directory
    if os.path.isdir(path):
        pass
    else:
        shutil.os.mkdir(path)
        
    #Makes Sub-Directory for Train and Test    
    for i in dir1:
        sub_path=os.path.join(path,i)
        if os.path.isdir(sub_path):
            pass
        else:
            shutil.os.mkdir(sub_path)
        
        shutil.os.mkdir(os.path.join(sub_path,label_dir))
        dir2.append(os.path.join(sub_path,label_dir))
       
    for file1 in train_set:
        img_path1 = os.path.join(source,file1)
        img1=Image.open(img_path1)
        img1.save(os.path.join(dir2[0],file1))
    
    for file2 in test_set:
        img_path2 = os.path.join(source,file2)
        img2=Image.open(img_path2)
        img2.save(os.path.join(dir2[1],file2))
        
    return dir2


# In[ ]:


def data_visualize(yes_dir,no_dir):
    
    print("The number of images with facemask in the training set labelled 'yes':",len(os.listdir(yes_dir[0])) )
    print("The number of images with facemask in the test set labelled 'yes':",len(os.listdir(yes_dir[1])))
    print("The number of images without facemask in the training set labelled 'no':",len(os.listdir(no_dir[0])))
    print("The number of images without facemask in the test set labelled 'no':",len(os.listdir(no_dir[1])))

