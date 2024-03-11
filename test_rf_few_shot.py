#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import numpy as np
import task_generator as tg
import argparse
from rf_classifier import RF

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-o","--ordered",type=bool,default=True)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
ORDERED = args.ordered

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()

    rf = RF()
    accuracies = []
    for i in range(TEST_EPISODE):
        total_rewards = 0
        counter = 0
        task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,5,15)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=5,split="train",shuffle=False)

        test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=5,split="test",shuffle=True) #true
        sample_images,sample_labels,support_names = sample_dataloader.__iter__().next()
        for test_images, test_labels, qry_names in test_dataloader:
            batch_size = test_labels.shape[0]

            sim_forest_rels = rf.get_batch_rels(support_names, qry_names, 5)

            _,predict_labels = torch.max(sim_forest_rels.data,1)

            rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

            total_rewards += np.sum(rewards)
            counter += batch_size

        accuracy = total_rewards/1.0/counter
        accuracies.append(accuracy)
    
    print("test accuracy:",sum(accuracies) / len(accuracies))

if __name__ == '__main__':
    main()
