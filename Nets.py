import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.autograd import Variable
import sys


class QueryAttention(nn.Module):

    def __init__(self):
        super(QueryAttention, self).__init__()

    def forward(self,input,query,len_inp):
        #input is b_size,num_seq,size
        #query is b_size,size
        #len_inp is array of b_size len
        att = torch.bmm(input,query.unsqueeze(-1)).squeeze(-1) / input.size(-1)
        try:
            out = self._masked_softmax(att,len_inp)
        except:
            print(input)
            print(query)
            sys.exit()
        return out
    
    def _masked_softmax(self,mat,len_s):
        len_s = torch.FloatTensor(len_s).type_as(mat.data).long()
        idxes = torch.arange(0,mat.size(1),out=mat.data.new(mat.size(1))).long().unsqueeze(0)
        mask = Variable((idxes<len_s.unsqueeze(1)).float(),requires_grad=False)
        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(-1,True)+0.0001

        ret = exp/sum_exp.expand_as(exp)

        a = (ret==ret)
        if torch.min(a).data[0] == 0:
            print(exp)
            print(sum_exp.expand_as(exp))
            print(mat)
            raise OverflowError()

        return ret




class UIAttentionalBiRNN3(nn.Module):

    def __init__(self, inp_size,ui_size, hid_size, dropout=0, RNN_cell=nn.LSTM):
        super(UIAttentionalBiRNN3, self).__init__()

        self.natt = hid_size*2
        self.rnn = RNN_cell(input_size=inp_size,hidden_size=hid_size,num_layers=1,bias=True,batch_first=True,dropout=dropout,bidirectional=True)
        self.lin = nn.Linear(hid_size*2,self.natt) #to_att_space
        self.ui2query = nn.Linear(ui_size*2,self.natt)
        self.qa = QueryAttention()
        self.param = nn.Parameter(torch.rand(self.natt))

        
    def forward(self, packed_batch,layer):
        
        rnn_sents,_ = self.rnn(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        
        enc_sents = F.tanh(enc_sents.transpose(0,1)) #b_size,seq,size
        genq = self.param.unsqueeze(0)
        biasq = self.ui2query(layer)
        q = F.tanh(genq+biasq)#F.tanh(self.ui2query(torch.cat([user_embs,item_embs],dim=-1))) #F.tanh(self.lin(F.tanh(user_embs*item_embs))) #


        att_sents = F.tanh(self.lin(enc_sents))
    
        res = self.qa(att_sents,q,len_s).unsqueeze(-1)
        
        summed =  torch.sum(enc_sents *  res,1)
        #print(F.tanh(user_embs*item_embs))
        #print(self.lin.weight.detach())
        #print(F.cosine_similarity(genq,biasq,dim=-1)+1)
        return summed, res, (q,att_sents)

        

###########################################################################################################

class HNAR(nn.Module):

    def __init__(self, ntoken, nusers, nitems, num_class, emb_size=200, hid_size=100):

        super(HNAR, self).__init__()
        self.emb_size = emb_size
        self.ui_size = 20
        self.embed = nn.Embedding(ntoken, emb_size,padding_idx=0)
        self.lin_out = nn.Linear(hid_size*2,num_class)
        self.lin0 = nn.Linear(self.ui_size*2,self.ui_size*2)
        self.lin1 = nn.Linear(self.ui_size*2,self.ui_size*2)
        self.lin2 = nn.Linear(self.ui_size*2,1)

        self.ub = nn.Embedding(nusers, 1)
        self.ib = nn.Embedding(nitems, 1)

        self.users = nn.Embedding(nusers, self.ui_size)
        I.normal(self.users.weight.data,0.01,0.01)
        self.items = nn.Embedding(nitems, self.ui_size)
        I.normal(self.items.weight.data,0.01,0.01)

        self.word = UIAttentionalBiRNN3(emb_size,self.ui_size, emb_size//2)
        self.sent = UIAttentionalBiRNN3(emb_size,self.ui_size, emb_size//2)

        self.lin_out.requires_grad = True
        self.temp = False
        self.register_buffer("mean",torch.Tensor(1))
        self.register_buffer("reviews",torch.Tensor())
        self.register_buffer("classes",torch.Tensor(range(num_class)))

        self.register_buffer("temp_value",Variable(torch.Tensor(1)))
        #self.temp_value = nn.Parameter(torch.ones(1))
        #self.temp_value.data[0] = 0.0
        self.param = nn.Parameter(torch.ones(1))
        self.class_bias = nn.Parameter(torch.rand(num_class))

    def set_emb_tensor(self,emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor
    
    def set_bias(self,mean,u_mean,i_mean):
        self.mean[0] = mean
        self.ub.weight.data = u_mean
        self.ib.weight.data = i_mean

    def _reorder_sent(self,sents,sent_order):
        
        sents = F.pad(sents,(0,0,1,0)) #adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0),sent_order.size(1),sents.size(1))

        return revs

    def forward(self, batch_reviews,users,items,sent_order,ui_indexs,ls,lr):
        
        u = users[ui_indexs]
        i = items[ui_indexs]

        emb_w = F.dropout(self.embed(batch_reviews),training=self.training)
        emb_u = self.users(u)
        emb_i = self.items(i)
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)


        to_w = F.tanh(self.lin0(torch.cat([emb_u,emb_i],dim=-1)))

        sent_embs , atw, qw = self.word(packed_sents,to_w)
        rev_embs = self._reorder_sent(sent_embs,sent_order)

        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lr,batch_first=True)

        emb_u2 = self.users(users)
        emb_i2 = self.items(items)

        a = F.tanh(self.lin0(torch.cat([emb_u2,emb_i2],dim=-1)))
        to_s = F.tanh(self.lin1(a))
        doc_embs, ats,_ = self.sent(packed_rev,to_s)

        # surrogate = F.normalize(emb_u+emb_i,dim=-1)
        real = doc_embs

        out = self.lin_out(real)
        pred_r = self.lin2(to_s)

   

        return out, pred_r , atw, ats,qw










###########################################################################################################

class HNARREG(nn.Module):

    def __init__(self, ntoken, nusers, nitems, num_class, emb_size=200, hid_size=100):

        super(HNARREG, self).__init__()
        self.num_class = num_class
        self.emb_size = emb_size
        self.ui_size = 20
        self.embed = nn.Embedding(ntoken, emb_size,padding_idx=0)
        self.lin_out = nn.Linear(hid_size*2,num_class)
        self.lin_out2 = nn.Linear(hid_size*2,1)

        self.lin0 = nn.Linear(self.ui_size*2,self.ui_size*2)
        self.lin1 = nn.Linear(self.ui_size*2,self.ui_size*2)
        self.lin2 = nn.Linear(self.ui_size*2,hid_size*2)

        

        self.users = nn.Embedding(nusers, self.ui_size,norm_type=2,max_norm=1)
        I.normal(self.users.weight.data,0.01,0.01)
        self.items = nn.Embedding(nitems, self.ui_size,norm_type=2,max_norm=1)
        I.normal(self.items.weight.data,0.01,0.01)

        self.word = UIAttentionalBiRNN3(emb_size,self.ui_size, emb_size//2)
        self.sent = UIAttentionalBiRNN3(emb_size,self.ui_size, emb_size//2)


        self.register_buffer("classes",torch.Tensor(range(num_class)))

        

    def set_emb_tensor(self,emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor


    def _reorder_sent(self,sents,sent_order):
        
        sents = F.pad(sents,(0,0,1,0)) #adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0),sent_order.size(1),sents.size(1))

        return revs

    def forward(self, batch_reviews,users,items,sent_order,ui_indexs,ls,lr):
        
        u = users[ui_indexs]
        i = items[ui_indexs]

        emb_w = F.dropout(self.embed(batch_reviews),training=self.training)
        emb_u = self.users(u)
        emb_i = self.items(i)
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)


        to_w = F.tanh(self.lin0(torch.cat([emb_u,emb_i],dim=-1)))

        sent_embs , atw = self.word(packed_sents,to_w)
        rev_embs = self._reorder_sent(sent_embs,sent_order)

        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lr,batch_first=True)

        emb_u2 = self.users(users)
        emb_i2 = self.items(items)
        inter = emb_u2 * emb_i2

        a = F.tanh(self.lin0(torch.cat([emb_u2,emb_i2],dim=-1)))
        to_s = F.tanh(self.lin1(a))
        doc_embs, ats = self.sent(packed_rev,to_s)

        real = F.normalize(doc_embs,dim=-1)
        

        out = self.lin_out(F.dropout(real,p=0.5,training=self.training))
        fake_doc = F.normalize(self.lin2(to_s),dim=-1)

        # print("---")
        # print(torch.cat([real[0,:6].unsqueeze(0),fake_doc[0,:6].unsqueeze(0)],dim=0))
        # print("---")


        return out, fake_doc ,real, self.lin_out2(fake_doc)

