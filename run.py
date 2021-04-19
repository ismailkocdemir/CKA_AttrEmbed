import os
import torch
from experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from models.resnet import ResnetConstants
from models.attribute_embeddings import AttributeEmbeddingsConstants
from dataset import *
import train
#import agg_results

parser.add_argument(
    '--run-name',
    type=str,
    required=True,
    help='run name [required]')


###################################
# DATASET
###################################
parser.add_argument(
    '--dataset-type',
    type=str,
    default='STL10',
    choices=['Cifar100', 'STL10'],
    help='which dataset to train on'
)
parser.add_argument(
    '--dataroot',
    type=str,
    default='./data',
    help='where dataset is located'
)
parser.add_argument(
    '--download-dataset',
    action='store_true',
    help='download the dataset from pytorch.datasets'   
)


###################################
# TRAINING PARAMS
###################################
parser.add_argument(
    '--num-epochs',
    default=200,
    type=int,
    help='Number of epochs to train'
)
parser.add_argument(
    '--batch-size',
    default=32,
    type=int,
    help='batch size.'
)
parser.add_argument(
    '--optimizer',
    type=str,
    default='SGD',
    choices=['SGD', 'Adam'],
    help='Optimizer. Choose from: [SGD, Adam]'
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-2,
    help='learning rate'
)
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum for the optimizer.'
)
parser.add_argument(
    '--num-workers',
    default=4,
    type=int,
    help='number of workers for the dataloader'
)


###################################
# MODEL
###################################
parser.add_argument(
    '--num-layers',
    default=18,
    type=int,
    choices=[18,34,50,101],
    help='Number of layer in the feature extractor(resnet): [18,34,50,101].'
)
parser.add_argument(
    '--sim-loss',
    action='store_true',
    help='include semantic similarity in the loss'
)
parser.add_argument(
    '--ce-loss-warmup',
    default=50,
    type=int,
    help='linearly increase the cross-entropy loss weight'
)
parser.add_argument(
    '--embed-type',
    type=str,
    default='vico_select',
    choices=['vico_linear', 'vico_select'],
    help='embedding types to be used in semantic similarity loss')
parser.add_argument(
    '--vico-dim',
    type=int, 
    default=200,
    help='dimension of embeddings if linear is selected')
parser.add_argument(
    '--hypernym',
    action='store_true',
    help='exclude hypernym co-oc. from vico embeddings')


##################################
#  LOGGING & SAVING
##################################
parser.add_argument(
    '--log-step',
    type=int,
    default=200,
    help='log at every log-step iterations.'
)
parser.add_argument(
    '--model-save-epoch',
    type=int,
    default=10,
    help='save the model at every model-save-step iterations.'
)
parser.add_argument(
    '--val-epoch',
    type=int,
    default=10,
    help='evaluate the model at every val-step iterations.'
)

def exp_train():
    args = parser.parse_args()
    
    # create experiments directory and required folders
    out_base_dir = os.path.join(
        os.getcwd(),
        f'exp/{args.dataset_type}'
    )
    exp_const = ExpConstants(args.run_name, out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    
    use_cuda = torch.cuda.is_available()
    exp_const.device = "cuda:0" if use_cuda else "cpu"
    
    # tranining params
    exp_const.optimizer = args.optimizer  
    exp_const.num_epochs = args.num_epochs
    exp_const.batch_size = args.batch_size
    exp_const.lr = args.lr 
    exp_const.momentum = args.momentum
    exp_const.num_workers = args.num_workers
    
    # logging, saving
    exp_const.log_step = args.log_step
    exp_const.model_save_epoch = args.model_save_epoch
    exp_const.val_epoch = args.val_epoch
    exp_const.subset = {
        'training': 'train',
        'test': 'test'
    }
    
    # dataset
    data_const = DatasetConstants(root=args.dataroot, download=args.download_dataset, train=True)
    data_const.dataset_type = args.dataset_type
    
    # model (resnet and attribute embeddings)
    model_const = Constants()
    model_const.model_num = None
    model_const.sim_loss = args.sim_loss
    model_const.ce_loss_warmup = args.ce_loss_warmup
    
    model_const.net = ResnetConstants()
    if args.dataset_type == 'Cifar100':
        model_const.net.num_layers = "cifar100" # a custom resnet for cifar100, to adjust the dimensions of the feature maps
        model_const.net.num_classes = 100 
    else:
        model_const.net.num_layers = args.num_layers
        if args.dataset_type == "Imagenet":
            model_const.net.num_classes = 1000
        elif args.dataset_type == "VOC":
            model_const.net.num_classes = 20
        elif args.dataset_type == "STL10":
            model_const.net.num_layers = 'cifar100' # TODO: deeper resnets does not work on STL10. 
            model_const.net.num_classes = 10

    model_const.net.pretrained = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
    
    model_const.attr_embed = AttributeEmbeddingsConstants()
    model_const.attr_embed_path = os.path.join(
        exp_const.model_dir,
        f'attr_embed_{model_const.model_num}')
    model_const.attr_embed.glove_dim = 300
    model_const.attr_embed.num_classes = model_const.net.num_classes

    # attribute embedding dimensions
    if args.embed_type=='vico_linear':
        model_const.attr_embed.no_glove = True # Zero out the glove component
        model_const.attr_embed.embed_dims = 300 + args.vico_dim
        embed_dir = os.path.join(
            os.getcwd(),
            'data/pretrained-embeddings/' + \
            f'glove_300_vico_linear_100/')
        model_const.attr_embed.embed_h5py = os.path.join(
            embed_dir,
            'visual_word_vecs.h5py')
        model_const.attr_embed.embed_word_to_idx_json = os.path.join(
            embed_dir,
            'visual_word_vecs_idx.json')
    elif args.embed_type=='vico_select':
        model_const.attr_embed.no_glove = True # Zero out the glove component
        model_const.attr_embed.hypernym = args.hypernym 
        model_const.attr_embed.embed_dims = 300 + args.vico_dim
        
        embed_dir = os.path.join(
            os.getcwd(),
            'data/pretrained-embeddings/' + \
            f'glove_300_vico_select_200/')
        model_const.attr_embed.embed_h5py = os.path.join(
            embed_dir,
            'visual_word_vecs.h5py')
        model_const.attr_embed.embed_word_to_idx_json = os.path.join(
            embed_dir,
            'visual_word_vecs_idx.json')
    else:
        err_str = f'{args.embed_type} is currently not implemented in the runner'
        assert(False), err_str

    # pass all constants to training method
    train.main(exp_const, data_const, model_const)

if __name__=='__main__':
    list_exps(globals())
