""" Main function for this repo. """
import argparse
import torch,json,os
import time
from misc import pprint,ensure_path,timer
from trainer.pre import PreTrainer
from trainer.adapt import AdaptTrainer
from trainer.self import SelfTrainer
import os.path as osp

# dataset_pool = ['BTNx','ACON_Ab','ACON_Ag','DeepBlue_Ag','RapidConnect_Ab','Quidel_Ag', 'Paramount_Ag', 'Abbot_Binax', 'AccessBio_Ag', 'Quidel_Ag_tight', 'abbott','oraquick','biomedomics','maxim','aytu','sd_igg']    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #### Basic parameters
    parser.add_argument('--n_class', type=int, default=2) # number of classes, how many classes in a task
    parser.add_argument('--phase', type=str, default='self_train', choices=['self_train', 'adapt_train', 'pre_train', 'self_test', 'adapt_test', 'pre_test']) # Phase
    parser.add_argument('--mode', type=str, default='self', choices=['self', 'adapt', 'pre']) # Mode
    parser.add_argument('--model_type', type=str, default='ResNet18ORI', choices=['ResNet18','ResNet18ORI']) # The network architecture
    # Configure optimizing
    parser.add_argument('--optim', type=str, default='Adam',choices=['Adam','SGD']) # Specify the optimzier
    parser.add_argument('--lr', type=float, default=0.001) # Learning rate, {0.001,0.0001,0.0005} for self-supervision and {0.001,0.0001} for adaptation
    parser.add_argument('--gamma', type=float, default=0.2) # learning rate decay value
    parser.add_argument('--step_size', type=int, default=30) # The number of epochs to for lr decay
    parser.add_argument('--batch_size', type=int, default=128) # Batch size
    parser.add_argument('--max_epoch', type=int, default=150) # maximum epoch number
    # Build dataset & dataloader
    parser.add_argument('--add_syn', type=int, default=1) # Add synthetic data into networkk training
    parser.add_argument('--train_all', type=int, default=1, choices=[0,1]) # Collect all available training data of all kit products
    parser.add_argument('--setname', type=str, default='BTNx')
    parser.add_argument('--label_mode', type=int, default=1) # The label file
    parser.add_argument('--dataset_dir', type=str, default='/home/ubuntu/Verify-SmartML/DATA/') # Dataset folder
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--noise_std', type=float, default=0) # the level of noise to be added during data augmentation
    parser.add_argument('--ctrl_full', type=float, default=0) #the ratio of control zones to be removed w.r.t. the number of total control zones of the target kit
    # Training & Test logic
    parser.add_argument('--log_root', type=str, default='./log/') # the root folder to save log
    parser.add_argument('--eval_weights', type=str, default=None) # Specify the model to be used (if necessary)
    parser.add_argument('--run', type=int, default=0) # Use the same parameter for multiple runs
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='1') # GPU id
    parser.add_argument('--white', type=int, default=0, help='white balancing during testing')


    #### Self-supervision parameters
    parser.add_argument('--xonly', type=int, default=0, choices=[0,1]) # Configure the edge filtering direction.
    parser.add_argument('--self_init', type=int, default=1) # Initialize the feature extractor with ImageNet pretrianed model or not.
    parser.add_argument('--cls_wgt', type=float, default=0) # weight of CE loss under self-supervision

    #### Few-shot adaptation parameters
    parser.add_argument('--color',type=int, default=1, choices=[0,1,2]) # Assign different data augmentation function
    parser.add_argument('--shot', type=int, default=10) # Number of training samples per class
    parser.add_argument('--adapt_init', type=int, default=10) # Specify the parameters of network initialization
    parser.add_argument('--ct_wgt', type=float, default=1.0)
    parser.add_argument('--conlayer',type=int, default=1, help='additional layer for contrastive learning')
    parser.add_argument('--supp_aug', type=int, default=2)
   

    # Set and print the parameters
    args = parser.parse_args()
    pprint(vars(args))
    args.mode = args.phase.split('_')[0]
    ensure_path(osp.join(args.log_root,args.mode))

    # Set manual seed for PyTorch
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if args.phase=='self_train':
        trainer = SelfTrainer(args)
        starttime = time.time()
        trainer.train()
        endtime = time.time()
        timer(endtime,starttime)
    elif args.phase=='self_test':
        pass
    elif args.phase=='pre_train':
        trainer = PreTrainer(args)
        trainer.train()
    elif args.phase=='pre_test':
        trainer = PreTrainer(args)
        trainer.eval('full')
    elif args.phase=='adapt_train':
        trainer = AdaptTrainer(args)
        trainer.train()
    elif args.phase == 'adapt_test':
        trainer = AdaptTrainer(args)
        model_root = './logs/adapt/'
        the_models = ['Quidel_Ag_ResNet18ORI_shot100_lr10.001_ct0.5_epo60_color2_init2_ctrl0.5_vis1_Adam_all','Quidel_Ag_ResNet18ORI_shot100_lr10.001_ct1.0_epo60_color2_init4_ctrl0.5_vis1_Adam_all']
        for the_model in the_models:
            _,_,msg = trainer.eval_epoch(0,os.path.join(model_root,the_model)) #'/home/jiawei/COVID/Eval_Zone_V3.0/logs/V300/meta_system/DeepBlue_Ag/')#
            print(msg,the_model)
    else:
        raise ValueError('Please set correct phase.')


    # parser.add_argument('--supp_aug', type=int, default=2) # Epoch number for meta-train phase
    # parser.add_argument('--max_epoch', type=int, default=60) # Epoch number for meta-train phase
    # parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    
    # parser.add_argument('--meta_lr1', type=float, default=0.001) # Learning rate for SS weights
    # parser.add_argument('--meta_lr2', type=float, default=0.001) # Learning rate for FC weights

    # parser.add_argument('--init_weights', type=str, default=None) # The pre-trained weights for meta-train phase
    # parser.add_argument('--meta_label', type=str, default='exp1') # Additional label for meta-train
    # parser.add_argument('--pre_stage', type=str, default='max_acc')
    
    # parser.add_argument('--visual_judge', type=int, default=1)
    # parser.add_argument('--distill_lr', type=float, default=0.001)
    # parser.add_argument('--student', type=int, default=0)
    # parser.add_argument('--noise_coe',type=float,default=0)
    # parser.add_argument('--conlayer', type=int, default=1)

    # parser.add_argument('--pre_max_epoch', type=int, default=150) # Epoch number for pre-train phase
    # parser.add_argument('--pre_batch_size', type=int, default=150) # Batch size for pre-train phase
    # parser.add_argument('--pre_lr', type=float, default=0.1) # Learning rate for pre-train phase
    # parser.add_argument('--pre_gamma', type=float, default=0.2) # Gamma for the pre-train learning rate decay
    # parser.add_argument('--pre_step_size', type=int, default=30) # The number of epochs to reduce the pre-train learning rate
    