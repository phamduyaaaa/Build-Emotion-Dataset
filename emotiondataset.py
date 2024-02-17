from torch.utils.data import Dataset
import os
import cv2
import pandas as pd

df = pd.read_csv('D:\\CoGangNhe\\Build Emotion Dataset\\data\\label\\label.lst', delimiter=' ', header=None)
class EmotionDataset(Dataset):
    def __init__(self,root):
        self.root = root
        self.all_img_paths = []
        self.all_filenames = []
        data_file = os.path.join(root)
        for filename in os.listdir(data_file):
            img_path = os.path.join(data_file, filename)
            self.all_img_paths.append(img_path)
            self.all_filenames.append(filename)
    def __len__(self):
        return len(self.all_img_paths)
    def __getitem__(self, index):
        img_path = self.all_img_paths[index]
        image = cv2.imread(img_path)
        name_image = df[0]
        filename = self.all_filenames[index]
        if filename in name_image.values:
            label = df[7][index]
        else:
            label = None
        return image, label
if __name__ == '__main__':
    train_dataset = EmotionDataset(root="data\\image\\origin")
    image,label = train_dataset[4564]
    cv2.imshow(str(label),image)
    cv2.waitKey(0)