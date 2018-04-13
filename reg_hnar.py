import argparse
import pickle as pkl
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Nets import HNARREG
from fmtl import FMTL
from utils import *
import quiviz
from quiviz.contrib import LinePlotObs
import sys
import numpy as np
from Data import BucketSampler
import logging

def save(net, dic, path):
    dict_m = net.state_dict()
    dict_m["word_dic"] = dic    
    torch.save(dict_m,path)

def tuple_batch(l):
    def min_one(rev):
        if len(rev)==0:
            rev = [[1]]
        return rev

    user, item, review,rating = zip(*l)
    r_t = torch.Tensor(rating).long()
    u_t = torch.Tensor(user).long()
    i_t = torch.Tensor(item).long()

    list_rev = [min_one(x) for x in review]
    

    sorted_r = sorted([(len(r),r_n,r) for r_n,r in enumerate(list_rev) ],reverse=True) #index by desc rev_le
    lr, r_n, ordered_list_rev = zip(*sorted_r)

    max_sents = lr[0]
    #reordered
    r_t = r_t[[r_n]]
    u_t = u_t[[r_n]]
    i_t = i_t[[r_n]]
    review = [review[x] for x in r_n] #reorder reviews

    stat =  sorted([(len(s), r_n, s_n, s) for r_n, r in enumerate(ordered_list_rev) for s_n, s in enumerate(r)], reverse=True)
    max_words = stat[0][0]
    ls = []
    batch_t = torch.zeros(len(stat), max_words).long()                          # (sents ordered by len)
    ui_indexs = torch.zeros(len(stat)).long()                                  # (sents original rev_n)
    sent_order = torch.zeros(len(ordered_list_rev), max_sents).long().fill_(0) # (rev_n,sent_n)

    for i,s in enumerate(stat):
        sent_order[s[1],s[2]] = i+1
        ui_indexs[i]=s[1]
        batch_t[i,0:len(s[3])] = torch.LongTensor(s[3])
        ls.append(s[0])

    return batch_t,r_t,u_t,i_t,sent_order,ui_indexs,ls,lr,review

def set_mean_tensors(data_gen,data_test,net):
    um = {}
    im = {}
    m = 0
    len_gen = 0
    ni = net.ib.weight.size(0)
    nu = net.ub.weight.size(0)


    for d in data_gen:
        user,item,_,rating = d
        #print(user)
        um.setdefault(user,[]).append(rating)
        im.setdefault(item,[]).append(rating)
        m += rating
        len_gen +=1

    m = m/len_gen
    print(m)

    nut = torch.Tensor(nu,1)
    for i in range(nu):
        if i in um:
            nut[i,0] = (sum(um[i])/len(um[i])) - m
        else:
            nut[i,0] = 0

    nit = torch.Tensor(ni,1)

    for i in range(ni):
        if i in im:
            nit[i,0] = (sum(im[i])/len(im[i])) - m
        else:
            nit[i,0] = 0

    net.set_bias(m,nut,nit)



@quiviz.log
def train(epoch,net,optimizer,dataset,criterion,cuda,optimize=False,msg="test"):
    if optimize:
        net.train()
    else:
        net.eval()

    epoch_loss = 0
    mean_mse = [0,0]
    mean_cos = 0
    ok_all = [0,0]
    d = 0
    rl = 0
    ce_loss = torch.nn.CrossEntropyLoss()

    ratings = Variable(torch.arange(0,5)).unsqueeze(0)
    
    if cuda:
        ratings = ratings.cuda()

    data_tensors = new_tensors(6,cuda,types={0:torch.LongTensor,1:torch.LongTensor,2:torch.LongTensor,3:torch.LongTensor,4:torch.LongTensor,5:torch.LongTensor}) #data-tensors

    with tqdm(total=len(dataset),desc=msg,ncols=70) as pbar:
        for iteration, (batch_t,r_t,u_t,i_t,sent_order,ui_indexs,ls,lr,review) in enumerate(dataset):
            data = tuple2var(data_tensors,(batch_t,r_t,u_t,i_t,sent_order,ui_indexs))

            d += data[1].size(0)

            if optimize:
                optimizer.zero_grad()

            sent ,fake_doc , real, r_reco = net(data[0],data[2],data[3],data[4],data[5],ls,lr)

            #r_pol = torch.sum(F.softmax(sent,dim=-1) * ratings,dim=-1)
            #r_reco = torch.sum(F.softmax(classified,dim=-1) * ratings,dim=-1)


            ok,per,_ = accuracy(sent,data[1])
            ok_all[0] += ok.data[0]

            mseloss_r = F.mse_loss(r_reco,data[1].float(),size_average=False)
            mean_mse[1] += mseloss_r.data[0]

            pred_loss_s =  criterion(sent, data[1])
            #pred_loss_r =  criterion(classified, data[1])

            mseloss_emb = torch.sum(torch.pow(fake_doc - real.detach(),2),dim=-1)
            
            loss = torch.mean(mseloss_emb) +  torch.mean(pred_loss_s) + torch.mean(mseloss_r) #+  0.01 * F.relu(reg-1.5) 

            rl += 0#reg.data[0] *0.01
            epoch_loss += loss.data[0]

            if optimize:
                loss.backward()
                optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix({"acc":round(ok_all[0]/d,4),"rmse":math.sqrt(round(mean_mse[1]/d,4)),"l":epoch_loss/d})

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, {}% accuracy".format(epoch,0,0))
    return {f"{msg}_acc":ok_all[0]/d,f"{msg}_rmse": math.sqrt(mean_mse[1]/d),f"{msg}_loss":epoch_loss/d}


def load(args):

    datadict = pkl.load(open(args.filename,"rb"))
    data_tl,(trainit,valit,testit) = FMTL_train_val_test(datadict["data"],datadict["splits"],args.split,validation=0.5,rows=datadict["rows"])


    if args.beer:
        rating_mapping = lambda x: max(0,int(float(x))-1)
        num_classes = 5
    else:
        rating_mapping = data_tl.get_field_dict("rating",key_iter=trainit) #creates class mapping
        rating_mapping = {v:i for i,v in enumerate(sorted(list(rating_mapping.keys()))) }
        #rating_mapping = {0.0:0,1.0:0,2.0:0,3.0:0,4.0:1,5.0:1}
        num_classes = len(set(rating_mapping.values()))
   
    data_tl.set_mapping("rating",rating_mapping,unk=-1) # if unknown #id is 0

    print(num_classes)

    user_mapping = data_tl.get_field_dict("user_id",key_iter=trainit,offset=1) #creates class mapping
    data_tl.set_mapping("user_id",user_mapping,unk=0) # if unknown #id is 0
    user_mapping["_unk_"] = 0

    item_mapping = data_tl.get_field_dict("item_id",key_iter=trainit,offset=1) #creates class mapping
    data_tl.set_mapping("item_id",item_mapping,unk=0)
    item_mapping["_unk_"] = 0

    if args.load:
        state = torch.load(args.load)
        wdict = state["word_dic"]
    else:
        if args.emb:
            tensor,wdict = load_embeddings(args.emb,offset=2)
        else:     
            wdict = data_tl.get_field_dict("review",key_iter=trainit,offset=2, max_count=args.max_feat, iter_func=(lambda x: (w for s in x for w in s )))

        wdict["_pad_"] = 0
        wdict["_unk_"] = 1
    
    if args.max_words > 0 and args.max_sents > 0:
        print("==> Limiting review and sentence length: ({} sents of {} words) ".format(args.max_sents,args.max_words))
        data_tl.set_mapping("review",(lambda f:[[wdict.get(w,1) for w in s ][:args.max_words] for s in f][:args.max_sents] ))
    else:
        data_tl.set_mapping("review",wdict,unk=1)

    print("Train set class stats:\n" + 10*"-")
    _,_ = data_tl.get_stats("rating",trainit,True)

    if args.load:
        net = HNARREG(ntoken=len(state["word_dic"]),nusers=state["users.weight"].size(0), nitems=state["items.weight"].size(0),emb_size=state["embed.weight"].size(1),hid_size=state["sent.rnn.weight_hh_l0"].size(1),num_class=state["lin_out.weight"].size(0))
        del state["word_dic"]
        net.load_state_dict(state)

    else:
        if args.emb:
            net = HNARREG(ntoken=len(wdict),nusers=len(user_mapping), nitems=len(item_mapping),emb_size=len(tensor[1]),hid_size=args.hid_size,num_class=num_classes)
            net.set_emb_tensor(torch.FloatTensor(tensor))
        else:
            net = HNARREG(ntoken=len(wdict),nusers=len(user_mapping), nitems=len(item_mapping), emb_size=args.emb_size,hid_size=args.hid_size, num_class=num_classes)

    if args.prebuild:
        data_tl = FMTL(list(x for x  in tqdm(data_tl,desc="prebuilding")),data_tl.rows)

    return data_tl,(trainit,valit,testit), net, wdict


def main(args):

    print(32*"-"+"\nHierarchical Neural Attention for Recommendation :\n" + 32*"-")
    data_tl, (train_set, val_set, test_set), net, wdict = load(args)

    dataloader = DataLoader(data_tl.indexed_iter(train_set), batch_size=args.b_size, shuffle=True, num_workers=3, collate_fn=tuple_batch, pin_memory=True)
    dataloader_valid = DataLoader(data_tl.indexed_iter(val_set), batch_size=args.b_size, shuffle=False,  num_workers=3, collate_fn=tuple_batch, pin_memory=True)
    dataloader_test = DataLoader(data_tl.indexed_iter(test_set), batch_size=args.b_size, shuffle=False, num_workers=3, collate_fn=tuple_batch, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss(size_average=False,reduce=False)      

    if args.cuda:
        net.cuda()

    print("-"*20)

    prec_val = 100
    lr = 0.1
    
    bl = []
    if args.emb:
        bl.append("embed.weight")

    optimizer = optim.Adam(param_blacklist(net,bl))
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)
    
    for epoch in range(1, args.epochs + 1):
        print("\n-------EPOCH {}-------".format(epoch))
        logging.info("\n-------EPOCH {}-------".format(epoch))
        train(epoch,net,optimizer,dataloader,criterion,args.cuda,optimize=True,msg="train")
        save_embs(net,"embtest",wdict)
        
        if args.snapshot:
            print("snapshot of model saved as {}".format(args.save+"_snapshot"))
            save(net,wdict,args.save+"_snapshot")

        val = train(epoch,net,optimizer,dataloader_valid,criterion,args.cuda,msg="val")
        if val['val_loss'] > prec_val:
            print("EARLY-STOPPING END")
            train(epoch,net,optimizer,dataloader_test,criterion,args.cuda,msg="test") 
            break
        prec_val = val['val_loss']
        train(epoch,net,optimizer,dataloader_test,criterion,args.cuda,msg="test")

    if args.save:
        print("model saved to {}".format(args.save))
        save(net,wdict,args.save)
        


def save_embs(net,path,wdict):
    ds = [{f'user-{i}':i for i in range(net.users.weight.size(0))},{f'item-{i}':i for i in range(net.items.weight.size(0))}]
    e_dict = {}
    
    for d in ds:
        for v,k in sorted([(v,k) for (k,v) in d.items()]):
            e_dict[k] = len(e_dict)



    all_embs = torch.cat((net.users.weight,net.items.weight),dim=0)
    
    save_embeddings(all_embs.cpu().data.numpy(),path,e_dict)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hierarchical Neural Attention for Recommendation')
    
    parser.add_argument("--emb-size",type=int,default=200)
    parser.add_argument("--hid-size",type=int,default=100)

    parser.add_argument("--max-feat", type=int,default=10000)
    parser.add_argument("--epochs", type=int,default=10)
    parser.add_argument("--clip-grad", type=float,default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--b-size", type=int, default=32)

    parser.add_argument("--emb", type=str)
    parser.add_argument("--max-words", type=int,default=-1)
    parser.add_argument("--max-sents",type=int,default=-1)

    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--snapshot", action='store_true')
    parser.add_argument("--vizdom", action='store_true')

    parser.add_argument("--prebuild",action="store_true")
    parser.add_argument('--beer', action='store_true')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    parser.add_argument("--output", type=str)
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    
    if args.vizdom:
        quiviz.register(LinePlotObs())
    quiviz.name_xp(f"{args.filename.split('/')[-1]}_{args.emb}_{args.split}")
    logging.info("========== NEW XP =============")
    
    try:
        main(args)
        pkl.dump({k:v for k,v in quiviz.quiviz._quiviz_shared_state.items()},open(f"{args.filename.split('/')[-1]}_{args.emb}_{args.split}.log.pkl","wb"))
    except Exception as e:
        pkl.dump({k:v for k,v in quiviz.quiviz._quiviz_shared_state.items()},open(f"{args.filename.split('/')[-1]}_{args.emb}_{args.split}.log.pkl","wb"))

        raise e
