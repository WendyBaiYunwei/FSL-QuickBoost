from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
import pickle
import json
import numpy as np
import torch
from config import Config
import time

class RF():
    def __init__(self):
        print('loading dataset')
        config = Config()
        with open(config.data_dir + 'rf_train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
            train_x = train_x.astype(np.float16)

        with open(config.data_dir + 'rf_train_y.pkl', 'rb') as f:
            train_y = pickle.load(f)
            train_y = train_y.astype(np.float16)

        with open(config.data_dir + 'embedding_resnet18_64classes_test.pkl', 'rb') as f:
            self.embeddings_test = pickle.load(f)
            self.embeddings_test = self.embeddings_test.astype(np.float16)

        with open(config.data_dir + 'imgNameToIdx_test.json', 'r') as f:
            self.name2idx_test = json.load(f)

        print(train_x.shape, train_y.shape)
        print('start RF training')
        start = time.time()
        self.classifier = RandomForestRegressor(n_estimators = 200, random_state = config.seed, max_features = 4)
        self.classifier.fit(train_x, train_y)
        end = time.time()
        print('done RF training. time taken is', end - start)
        del train_x
        del train_y

    def get_batch_rels(self, support_names, qry_names, shot_size = 1):
        relations = []
        spt_embs = [] # support embeddings

        classpt_embs = []
        embeddings = []
        names = support_names.copy()
        names.extend(qry_names)
        for name in names:
            name = name.split('/')[-1]
            embedding = self.embeddings_test[self.name2idx_test[name]]
            embeddings.append(embedding)
        embeddings = np.array(embeddings)

        ssEmbeddings = embeddings[:shot_size*5,:].reshape(shot_size, 5, -1)
        classEmbeddings = np.mean(ssEmbeddings, axis = 0).reshape(5, -1)

        qEmbeddings = embeddings[shot_size*5:,:]
        for i in range(len(qEmbeddings)):
            qEmbedding = qEmbeddings[i]
            for classI in range(5):
                avgSembedding = classEmbeddings[classI]
                diff = (avgSembedding - qEmbedding) ** 2
                relations.append(diff)    
    
        relations = np.stack(relations).reshape(-1, embeddings.shape[-1])
        preds = self.classifier.predict(relations)
        preds = preds.reshape(len(qEmbeddings), -1)
        return torch.from_numpy(preds)
