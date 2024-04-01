from torch.utils.data import Dataset

from MyResnet import preprocess

class BreastDataSet(Dataset):
    #We must implement init, len and getitem
    def __init__(self,imgs, labels, transform = None) -> None:
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]

        #Resize image to fixed size
        new_size = (256, 256)
        img.resize(new_size)

        #transform images
        if self.transform != None:
            transformed_img = self.transform(img)
            #transformed_img = preprocess(img)

        #Get label of img
        label = self.labels[idx]

        return transformed_img, label
