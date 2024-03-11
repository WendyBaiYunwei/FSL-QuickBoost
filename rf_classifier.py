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
        # self.classifier = ExtraTreesRegressor(n_estimators = 200, random_state = 0)
        self.classifier.fit(train_x, train_y)
        end = time.time()
        print('done RF training. time taken is', end - start)
        del train_x
        del train_y

    def get_batch_rels(self, support_names, qry_names, shot_size = 1):
        relations = []
        spt_embs = [] # support embeddings
        classpt_embs = []
        for i, support_name in enumerate(support_names):
            if i % shot_size == 0 and i > 0:
                avg_spt_emb = np.mean(spt_embs, axis = 0)
                classpt_embs.append(avg_spt_emb)
                spt_embs.clear()
            tokens = support_name.split('/')
            support_name = tokens[-1]
            spt_embedding = self.embeddings_test[self.name2idx_test[support_name]]
            spt_embs.append(spt_embedding)
        avg_spt_emb = np.mean(spt_embs, axis = 0)
        classpt_embs.append(avg_spt_emb)
        spt_embs.clear()

        for qry_name in qry_names:
            tokens = qry_name.split('/')
            qry_name = tokens[-1]
            qry_emb = self.embeddings_test[self.name2idx_test[qry_name]]
            for classI in range(5):
                avg_spt_emb = classpt_embs[classI]
                diff = (avg_spt_emb - qry_emb) ** 2
                relations.append(diff)    
        relations = np.stack(relations).reshape(-1, 512)
        preds = self.classifier.predict(relations)
        preds = preds.reshape(len(qry_names), -1)
        return torch.from_numpy(preds)