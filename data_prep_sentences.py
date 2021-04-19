'''
Return IMDB data as a list of sentences and associated labels
1 = positive
0 = negative
'''
import scandir

_DESCRIPTION = """\
Large Movie Review Dataset.
This is a dataset for binary sentiment classification containing substantially \
more data than previous benchmark datasets. We provide a set of 25,000 highly \
polar movie reviews for training, and 25,000 for testing. There is additional \
unlabeled data for use as well.\
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

_DOWNLOAD_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def get_reviews(dir):
    review_files = [f.name for f in scandir.scandir(dir)]
    review_list = []
    for review_file in review_files:
        with open(dir+'/'+review_file, "r", encoding="utf8") as f:
            text = f.read()
            text = text.rstrip('\n')
        review_list.append(text)
    return review_list

def get_data(base_dir):

    neg = base_dir + '/neg'
    pos = base_dir + '/pos'

    neg_review_list = get_reviews(neg)
    pos_review_list = get_reviews(pos)

    neg_labels = [0]*len(neg_review_list)
    pos_labels = [1]*len(pos_review_list)

    return neg_review_list, pos_review_list, neg_labels, pos_labels

def get_train(arch, base_dir=None):
    if base_dir == None:
        base_dir = '../data/train'
    else:
        base_dir = base_dir + 'train'
    return get_data(base_dir, arch)

def get_test(arch, base_dir=None):
    if base_dir == None:
        base_dir = '../data/test'
    else:
        base_dir = base_dir + 'test'
    return get_data(base_dir, arch)
