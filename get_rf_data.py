import random
import pickle
import numpy as np
from config import Config

class RF_data():
    def get_data(self, class_start, class_end):
        sample_size = 50
        iterations = 2
        IMGS_PER_CLASS = 600
        EMBEDDING_SIZE = 512 # resnet 18 has output embedding size of 512
        x_list = []
        y_list = []
        full_indices = set([i for i in range(class_start * IMGS_PER_CLASS, class_end * IMGS_PER_CLASS)])
        for class_idx in range(class_start, class_end):
            same_class_idxs = set([i for i in range(class_idx * IMGS_PER_CLASS, (class_idx + 1) * IMGS_PER_CLASS)])
            diff_class_idxs = full_indices - same_class_idxs
            for i in range(iterations):
                embedding_idx = class_idx * IMGS_PER_CLASS + i
                current_image = self.embeddings[embedding_idx]
                current_image = current_image / np.linalg.norm(current_image) 
                cur = np.expand_dims(current_image, axis = 0)
                cur = np.repeat(cur, sample_size, 0)
                
                # get same class images' indices
                same_class_imgs = random.sample(same_class_idxs, k = sample_size)
                
                same_class_embs = []
                for img_i in same_class_imgs:
                    same_class_emb = self.embeddings[img_i]
                    same_class_emb = same_class_emb / np.linalg.norm(same_class_emb) 
                    same_class_embs.append(same_class_emb)
                same_class_imgs = np.stack(same_class_embs).squeeze()
                diff = (cur - same_class_imgs) ** 2
                x_list.append(diff)
                # label same class image pairs as '1'
                labels = np.repeat(np.ones(1), sample_size, 0)
                y_list.append(labels)

                # get different class images' indices
                diff_class_idxs = random.sample(diff_class_idxs, k = sample_size)
                diff_img_embs = self.embeddings[diff_class_idxs].squeeze()
                cosine_sims = []
                for diff_i in range(len(diff_img_embs)):
                    diff_img_embs[diff_i] =\
                        diff_img_embs[diff_i] / np.linalg.norm(diff_img_embs[diff_i]) 
                diff = (cur - diff_img_embs) ** 2 
                x_list.append(diff)
                # label same class image pairs as '0'
                labels = np.repeat(np.zeros(1), sample_size, 0)
                y_list.append(labels)

        x_list = np.stack(x_list).reshape(-1, EMBEDDING_SIZE)
        y_list = np.stack(y_list).reshape(-1, 1)

        return x_list, y_list

    def __init__(self):
        config = Config()
        random.seed(config.seed)
        with open(config.data_dir + 'embedding_resnet18_64classes.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)

        CLASS_SIZE = 64 # miniImagenet has 64 classes for training

        trainX, trainY = self.get_data(0, CLASS_SIZE)

        with open(config.data_dir + 'rf_train_x.pkl', 'wb') as f:
            pickle.dump(trainX, f)

        with open(config.data_dir + 'rf_train_y.pkl', 'wb') as f:
            pickle.dump(trainY, f)

        print('QuickBoost data successfully created .')
        print(trainX.shape, trainY.shape)

RF_data()
