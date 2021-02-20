import sys
import copy
import torch
import torch.nn as nn
import numpy as np

sys.path.append("..")
from utils import io

class AttributeEmbeddingsConstants(io.JsonSerializableClass):
    def __init__(self):
        super(AttributeEmbeddingsConstants,self).__init__()
        self.num_classes = 100
        self.glove_dim = 300
        self.embed_dims = 500
        self.embed_h5py = None
        self.embed_word_to_idx_json = None
        self.no_glove = True
        self.hypernym = False

class AttributeEmbeddings(nn.Module):
    def __init__(self,const):
        super(AttributeEmbeddings,self).__init__()
        self.const = copy.deepcopy(const)
        #"""
        self.embed = nn.Embedding(
            self.const.num_classes,
            self.const.num_classes).requires_grad_(False)
        #"""
        
        # ABLATION WITH IDENTITY MATRIX INSTEAD OF EMBEDDING SIMILARITY MATRIX
        #self.embed = torch.eye(self.const.num_classes).cuda().requires_grad_(False)
        

    def load_embeddings(self,labels):
        #"""
        embed_h5py = io.load_h5py_object(self.const.embed_h5py)['embeddings']
        word_to_idx = io.load_json_object(self.const.embed_word_to_idx_json)
        embeddings = np.zeros([len(labels),self.const.embed_dims])
        word_to_label = {}
        for i,label in enumerate(labels):
            if ' ' in label:
                words = label.split(' ')
            elif '_' in label:
                words = label.split('_')
            else:
                words = [label]

            denom = len(words)
            for word in words:
                if word=='tree':
                    denom = len(words)-1
                    continue

                if word not in word_to_label:
                    word_to_label[word] = set()
                word_to_label[word].add(label)

                idx = word_to_idx[word]
                embeddings[i] += embed_h5py[idx][()]
            embeddings[i] /= denom

        if self.const.no_glove:
            embeddings[:,:self.const.glove_dim] = 0
        if self.const.hypernym == False:
            embeddings[:,self.const.glove_dim+100:self.const.glove_dim+150] = 0
        
        self.embed.weight.data.copy_(torch.from_numpy(embeddings))
        self.embed.weight -= torch.mean(self.embed.weight, dim=0, keepdims=True)
        #"""
        
        # ABLATION WITH IDENTITY MATRIX INSTEAD OF EMBEDDING SIMILARITY MATRIX
        #self.embed -= torch.mean(self.embed, dim=0, keepdims=True)

    def forward(self, feats, label_idxs, target):
        feats = self.pool_feats(feats, self.const.pool_size)
        feats = feats - torch.mean(feats, dim=0, keepdims=True)
        
        embed = torch.index_select(self.embed.weight, 0, label_idxs)
        #embed = torch.index_select(self.embed, 0, label_idxs)
        #embed = embed - torch.mean(embed, dim=0, keepdims=True)

        class_sim = self.center_gram(self.gram_linear(feats))
        embed_sim = self.center_gram(self.gram_linear(embed))  

        embed_norm = torch.norm(embed_sim, dim=(0,1))
        class_norm = torch.norm(class_sim, dim=(0,1))
        cka = torch.sum((class_sim*embed_sim))/(class_norm*embed_norm)

        """
        DEBUGGING FOR DIFFERENT SIMILARITY TARGETS
        with torch.no_grad():
            cka2 = torch.sum((class_sim/class_norm)*(embed_sim/embed_norm))
            l2_loss = torch.sum(((class_sim/class_norm) - (embed_sim/embed_norm))**2)
            print("Normal CKA", cka)
            print("CKA_v2", cka2)
            print("2-2CKA_v2", 2*(1-cka2))
            print("SymNMF", l2_loss)
        """

        return cka, torch.max(torch.cuda.FloatTensor([0]), target-cka)**2

    def pool_feats(self, feats, pooling_size):
        if len(feats.shape) > 2:
            feats = nn.functional.adaptive_avg_pool2d(feats, max_size)
        return feats.reshape(feats.shape[0], -1)

    def gram_linear(self, x):
        return torch.mm(x,x.t())

    def center_gram(self, gram):
        n = gram.shape[0]        
        gram = self.subtract_diag(gram)
        means = torch.sum(gram, dim=0) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram = self.subtract_diag(gram)
        return gram

    def subtract_diag(self, gram):
        diag_elements = torch.diag_embed(torch.diagonal(gram, 0))
        gram -= diag_elements
        return gram
