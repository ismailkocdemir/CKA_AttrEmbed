import os

from experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from models.resnet import ResnetConstants
from models.attribute_embeddings import AttributeEmbeddingsConstants
from dataset import Cifar100DatasetConstants
import train
import agg_results

parser.add_argument(
    '--embed_type',
    type=str,
    choices=['vico_linear', 'vico_select'],
    help='embedding types')
parser.add_argument(
    '--vico_dim',
    type=int, default=200,
    help='dimension of embeddings if linear is selected')
parser.add_argument(
    '--hypernym',
    action='store_true',
    help='exclude hypernym co-oc. from vico embeddings')
parser.add_argument(
    '--sim-loss',
    action='store_true',
    help='include semantic similarity in the loss')
parser.add_argument(
    '--run',
    type=int,
    default=0,
    help='Run number')

def exp_train():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'embed_type',
            'run'],
        optional_args=[])
    
    ext_simloss = "w-hypernym" if args.hypernym else "wo-hypernym"
    ext_hypernym = "w-simloss" if args.sim_loss else "wo-simloss"
    exp_name = \
        args.embed_type + '_' + \
        ext_hypernym + '_' + \
        ext_simloss
    out_base_dir = os.path.join(
        os.getcwd(),
        f'exp/cifar100/run_{args.run}')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 200
    exp_const.model_save_step = 1000
    exp_const.val_step = 1000
    exp_const.batch_size = 128
    exp_const.num_epochs = 200
    exp_const.lr = 0.1 
    exp_const.momentum = 0.9
    exp_const.num_workers = 1
    exp_const.optimizer = 'SGD'
    exp_const.subset = {
        'training': 'train',
        'test': 'test'
    }

    data_const = Cifar100DatasetConstants()
    
    model_const = Constants()
    model_const.model_num = None
    model_const.sim_loss = args.sim_loss
    model_const.net = ResnetConstants()
    
    #Just for Cifar100. This should be an integer otherwise.
    model_const.net.num_layers = "cifar100" 
    
    model_const.net.num_classes = 100
    model_const.net.pretrained = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
    model_const.attr_embed = AttributeEmbeddingsConstants()
    model_const.attr_embed_path = os.path.join(
        exp_const.model_dir,
        f'attr_embed_{model_const.model_num}')
    model_const.attr_embed.glove_dim = 300

    # Dimensions
    if args.embed_type=='vico_linear':
        model_const.attr_embed.no_glove = True # Zero out the glove component
        model_const.attr_embed.embed_dims = 300 + args.vico_dim
        embed_dir = os.path.join(
            os.getcwd(),
            'data/pretrained-embeddings/' + \
            f'glove_300_linear_100/')
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

    train.main(exp_const,data_const,model_const)

if __name__=='__main__':
    list_exps(globals())
