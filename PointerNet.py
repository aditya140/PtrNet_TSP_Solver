import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self,emb_dim,hid_dim,num_layers,bidir,dropout):
        super().__init__()
        self.hid_dim=hid_dim//2 if bidir else hid_dim
        self.num_layers=num_layers*2 if bidir else num_layers
        self.emb_dim=emb_dim
        self.embedding=nn.Linear(2,emb_dim)
        self.rnn=nn.LSTM(self.emb_dim,self.hid_dim,num_layers,bidirectional=bidir,dropout=dropout)
        self.h0=nn.Parameter(torch.zeros(1),requires_grad=False)

    def forward(self,inp):
        embedded=self.embedding(inp)
        h=self.h0.unsqueeze(0).unsqueeze(0).repeat(self.num_layers,embedded.shape[0],self.hid_dim)
        c=self.h0.unsqueeze(0).unsqueeze(0).repeat(self.num_layers,embedded.shape[0],self.hid_dim)
        embedded=embedded.permute(1,0,2)
        opt,hid=self.rnn(embedded,(h,c))
        return opt.permute(1,0,2),hid,embedded.permute(1,0,2)


class Attention(nn.Module):
    def __init__(self,inp_dim,hid_dim):
        super().__init__()
        self.inp_dim=inp_dim
        self.hid_dim=hid_dim
        self.inp_layer=nn.Linear(inp_dim,hid_dim)
        self.context_layer=nn.Conv1d(inp_dim,hid_dim,1,1)
        self._inf=nn.Parameter(torch.FloatTensor([float('-inf')]),requires_grad=False)
        self.V=nn.Parameter(torch.FloatTensor(hid_dim),requires_grad=True)

        nn.init.uniform_(self.V,-1,1)
    def forward(self,inp,context,mask):
        """
        Param : inp [batch_size, hid_dim]
        Param : context [batch_size, seq_len , hid_dim]
        Param : mask [batch_size, seq_len])]
        """
        hid = self.inp_layer(inp)
        hid = hid.unsqueeze(2).expand(-1, -1, context.size(1))
        context = context.permute(0, 2, 1)
        ctx = self.context_layer(context)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)
        att = torch.bmm(V, torch.tanh(hid + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = F.softmax(att,dim=1)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)
        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

class Decoder(nn.Module):
    def __init__(self,emb_dim,hid_dim):
        super().__init__()
        self.emb_dim=emb_dim
        self.hid_dim=hid_dim
        self.hid_out=nn.Linear(hid_dim*2,hid_dim)
        self.rnn=nn.LSTMCell(emb_dim,hid_dim)
        self.att=Attention(hid_dim,hid_dim)

        self.mask=nn.Parameter(torch.ones(1),requires_grad=False)
        self.runner=nn.Parameter(torch.ones(1),requires_grad=False)
    def forward(self,emb_inp,dec_inp,hid,context):
        batch_size=emb_inp.shape[0]
        inp_len=emb_inp.shape[1]
        mask = self.mask.repeat(inp_len).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())
        runner=self.runner.repeat(inp_len)
        for i in range(inp_len):
            runner.data[i]=i
        runner=runner.unsqueeze(0).expand(batch_size,-1).long()
        out=[]
        pointers=[]
        for i in range(inp_len):
            h_t,c_t=self.rnn(dec_inp,hid)
            h_t, outs = self.att(h_t, context, torch.eq(mask, 0)) # (batch, hid) (batch,seq_len)
            h_t = torch.tanh(self.hid_out(torch.cat((h_t, h_t), 1)))
            hid = (h_t,c_t)
            masked_outs = outs * mask
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()
            mask  = mask * (1 - one_hot_pointers)
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.emb_dim).byte()
            decoder_input = emb_inp[embedding_mask.data.bool()].view(batch_size, self.emb_dim)
            out.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))
        outputs = torch.cat(out).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hid


class PointerNet(nn.Module):    
    def __init__(self,emb_dim,hid_dim,num_layers,bidir,dropout):
        """[summary]
        PointerNet Module
        Arguments:
            emb_dim {[int]} -- Embedding Dimension
            hid_dim {[int]} -- Hidden Dimension
            num_layers {[int]} -- Num Layers
            bidir {[bool]} -- Bidirectional
            dropout {[float]} -- Dropout value
        """        
        super().__init__()
        self.bidir=bidir
        self.encoder=Encoder(emb_dim,hid_dim,num_layers,bidir,dropout,)
        self.h0=nn.Parameter(torch.FloatTensor(emb_dim),requires_grad=False)
        self.decoder=Decoder(emb_dim,hid_dim)
        nn.init.uniform_(self.h0, -1, 1)
    def forward(self,inp):
        batch_size=inp.shape[0]
        enc_out,enc_hid,emb_inp=self.encoder(inp)
        dec_inp0=self.h0.unsqueeze(0).expand(batch_size,-1)
        if self.bidir:
            dec_hid0 = (enc_hid[0][-2:],
                              enc_hid[1][-2:])
        else:
            dec_hid0 = (enc_hid[0][-1],
                               enc_hid[1][-1])
        
        (outputs, pointers), decoder_hidden = self.decoder(emb_inp,
                                                           dec_inp0,
                                                           dec_hid0,
                                                           enc_out)
        return  outputs, pointers
        
