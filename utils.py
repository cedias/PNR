#utils.py
import torch
from tqdm import tqdm
import collections
from operator import itemgetter
from torch.autograd import Variable
import numpy as np
from fmtl import FMTL


def param_blacklist(net,exclusion_list=[]):
    return [v for k,v in net.named_parameters() if k not in exclusion_list]

def param_whitelist(net,param_list=[]):
    return [v for k,v in net.named_parameters() if k in param_list]
            
def tuple2var(tensors,data):
    def copy2tensor(t,data):
        t.resize_(data.size()).copy_(data,async=True)
        return Variable(t)
    return tuple(map(copy2tensor,tensors,data))

def new_tensors(n,cuda,types={}):
    def new_tensor(t_type,cuda):
        x = torch.Tensor()
        if t_type:
            x = x.type(t_type)
        if cuda:
            x = x.cuda()
        return x
    return tuple([new_tensor(types.setdefault(i,None),cuda) for i in range(0,n)])


def accuracy(out,truth):
    def sm(mat):
        exp = torch.exp(mat)
        sum_exp = exp.sum(1,True)+0.0001
        return exp/sum_exp.expand_as(exp)

    _,max_i = torch.max(sm(out),1)

    eq = torch.eq(max_i,truth).float()
    all_eq = torch.sum(eq)

    return all_eq, all_eq/truth.size(0)*100, max_i.float()

def checkpoint(epoch,net,output):
    model_out_path = output+"_epoch_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def load_embeddings(file,offset=0):
    emb_file = open(file).readlines()
    first = emb_file[0]
    word, vec = int(first.split()[0]),int(first.split()[1])
    size = (word,vec)
    print("--> Got {} words of {} dimensions".format(size[0],size[1]))
    tensor = torch.zeros(size[0]+offset,size[1]) ## adding offset
    word_d = {}

    print("--> Shape with padding and unk_token:")
    print(tensor.size())

    for i,line in tqdm(enumerate(emb_file),desc="Creating embedding tensor",total=len(emb_file)):
        if i==0: #skipping header (-1 to the enumeration to take it into account)
            print("skipping embedding size line:\'{}\'".format(line.strip()))
            continue

        spl = line.strip().split(" ")

        if len(spl[1:]) == size[1]: #word is most probably whitespace or junk if badly parsed
            word_d[spl[0]] = i + offset-1
            tensor[i+offset-1] = torch.FloatTensor(list(float(x) for x in spl[1:]))
        else:
            print("WARNING: MALFORMED EMBEDDING DICTIONNARY:\n {} \n line isn't parsed correctly".format(line))

    try:
        assert(len(word_d)==size[0])
    except:
        print("Final dictionnary length differs from number of embeddings - some lines were malformed.")

    return tensor, word_d



def FMTL_train_val_test(datatuples,splits,split_num=0,validation=0.5,rows=None):
    """
    Builds train/val/test indexes sets from tuple list and split list
    Validation set at 0.5 if n split is 5 gives an 80:10:10 split as usually used.
    """
    train,test = [],[]

    for idx,split in tqdm(enumerate(splits),total=len(splits),desc="Building train/test of split #{}".format(split_num)):
        if split == split_num:
            test.append(idx)
        else:
            train.append(idx)

    if len(test) <= 0:
            raise IndexError("Test set is empty - split {} probably doesn't exist".format(split_num))

    if rows and type(rows) is tuple:
        rows = {v:k for k,v in enumerate(rows)}
        print("Tuples rows are the following:")
        print(rows)

    if validation > 0:

        if 0 < validation < 1:
            val_len = int(validation * len(test))

        validation = test[-val_len:]
        test = test[:-val_len]

    else:
        validation = []

    idxs = (train,test,validation)
    fmtl = FMTL(datatuples,rows)
    iters = idxs

    return (fmtl,iters)


def groupby(key, seq):
    """ Group a collection by a key function

    >>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    >>> groupby(len, names)  # doctest: +SKIP
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}
    """
    if not callable(key):
        key = getter(key)
    d = collections.defaultdict(lambda: [].append)
    for item in seq:
        d[key(item)](item)
    rv = {}
    for k, v in d.items():
        rv[k] = v.__self__
    return rv


def get_other_revs(ig, max_revs):
    def mp1(r):
        if r == 2:
            return 0
        elif r<2:
            return -1
        else:
            return 1
    def filter_transform(t):
        #tuple (u,i,t,r) function
        user = t[0]
        item = t[1]
        #gets other reviews.
        txt_u = [(u,r) for u, _, t, r in ig.get(item, [])[:max_revs] if u != user]
        txt_u.append((0,2))
        t.append(txt_u)

        return t

    return filter_transform


def save_embeddings(embs,path,dict_label=None):
    """
    embs is Numpy.array(N,size)
    dict_label is {str(word)->int(idx)} or {int(idx)->str(word)}
    """
    def int_first(k,v):
        if type(k) == int:
            return (k,v)
        else:
            return (v,k)

    np.savetxt(f"{path}_vectors.tsv", embs, delimiter="\t")

    #labels 
    if dict_label:
        sorted_labs = np.array([lab for idx,lab in sorted([int_first(k,v) for k,v in dict_label.items()])])
        print(sorted_labs)
        with open(f"{path}_metadata.tsv","w") as metadata_file:
            for x in sorted_labs: #hack for space
                if len(x.strip()) == 0:
                    x = f"space-{len(x)}"
                    
                metadata_file.write(f"{x}\n")
    #np.savetxt("label.tsv", target.cpu().numpy(), delimiter="\t")
