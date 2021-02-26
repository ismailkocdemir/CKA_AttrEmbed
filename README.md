### Centered Kernel Alignment on Attribute Embeddings

In brief, I want to align the similarity kernel (Kornblith et. al, 2019) of image features (from CNNs) between categories with the similarity kernel of the word embeddings for the categories (adapted from ViCo (Gupta et. al, 2019), but there are many other options for language models: BERT, word2vec, GloVe). 

Given a supervised image classification task, my aim is to use the language supervision by trying to align the kernels representing the inter-similarities between the target categories induced by two different types of representational spaces: visual features and word embeddings. To that end, there are two aspects worth noting: 

1. Mapping of each category to the word embedding space with their mere name or description might not always be helpful. I specifically look for an attribute set consisting of visually descriptive words (e.g. attributes collected in the Visual Genome Dataset). Having this set, we can represent each category by projecting its label onto this attribute set in the embedding space, effectively forming a representation of the category that is more refined in terms of visual perception.

2. Aligning the similarity kernels from the two spaces saves us from learning a new mapping from visual representations to word embedding, or a new shared embedding space.




#### References

Kornblith, Simon, et al. "Similarity of neural network representations revisited." International Conference on Machine Learning. PMLR, 2019.

Gupta, Tanmay, Alexander Schwing, and Derek Hoiem. "ViCo: Word embeddings from visual co-occurrences." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.











