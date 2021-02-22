### Centered Kernel Alignment on Attribute Embeddings

In brief, I want to align the similarity kernel (Kornblith et. al, 2019) of image features (from CNNs) between categories with the similarity kernel of the word embeddings for the categories (adapted from ViCo (Gupta et. al, 2019)). 

Two aspects worth more details regarding the methodology:

1. Any other of word embeddings could also be used apart from ViCo (BERT, word2vec, GloVe), however mapping of each category to the words embeddings space with their mere name is not always helpful. I specifically look for an attribute set consisting of visually descriptive words (e.g. attributes collected in the Visual Genome Dataset). Having this set, we can represent each category by projecting its name onto this attribute set in the embedding space, effectively forming a representation of the category that is more refined in terms of visual perception.

2. Rather than optimizing for a direct match between the image features of a category to the its representation in word emmbeddings, I'd like to impose the similarity structure between the categories in the embedding space revealed by the approach metioned in point 1, onto the similarity kernel between the image features from different categories.  



#### References

Kornblith, Simon, et al. "Similarity of neural network representations revisited." International Conference on Machine Learning. PMLR, 2019.

Gupta, Tanmay, Alexander Schwing, and Derek Hoiem. "ViCo: Word embeddings from visual co-occurrences." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.











