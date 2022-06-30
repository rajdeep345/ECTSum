import gc
import torch
from torch import nn
from torch.nn import init
from transformers import BertModel, RobertaModel

class MatchSum(nn.Module):
    
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        
        self.hidden_size = hidden_size
        self.candidate_num  = candidate_num
        
        if encoder == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, text_id, candidate_id, summary_id):
        
        batch_size = text_id.size(0)
        
        pad_id = 0     # for BERT
        if text_id[0][0] == 0:
            pad_id = 1 # for RoBERTa

        # get document embedding
        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0] # last layer
        doc_emb = out[:, 0, :]
        assert doc_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]
        
        # get summary embedding
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0] # last layer
        summary_emb = out[:, 0, :]
        assert summary_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

        # get summary score
        summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)
        
        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1) # [batch_size, candidate_num]
        assert score.size() == (batch_size, candidate_num)

        return {'score': score, 'summary_score': summary_score}


# #Chunking Implementation
# class MatchSum(nn.Module):
    
#     def __init__(self, candidate_num, encoder, hidden_size=768):
#         super(MatchSum, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.candidate_num  = candidate_num
        
#         if encoder == 'bert':
#             self.encoder = BertModel.from_pretrained('bert-base-uncased')
#         else:
#             self.encoder = RobertaModel.from_pretrained('roberta-base')

#     def forward(self, text_id, candidate_id, summary_id):
#         #text_id is chunked --> [bs, n_chunks, max_len] --> [1, 8, 512]
        
#         batch_size = text_id.size(0)
#         chunk_size = text_id.size(1)
#         pad_id = 0     # for BERT
#         if text_id[0][0][0] == 0:
#             pad_id = 1 # for RoBERTa
#         #-----------------------------------------------------------------------------------

#         # Get document embedding 
        
#         # Get BERT encoded representations of every chunk (This implementation gets cuda error)
#         # outs_list = []
#         # for i in range(chunk_size):
#         #     chunk_id = text_id[:,i,:]
#         #     input_mask = ~(chunk_id == pad_id)
#         #     outs_list.append(self.encoder(chunk_id, attention_mask=input_mask)[0].squeeze(0)) # last layer
#         #     del chunk_id, input_mask
#         #     gc.collect()        
#         # out = torch.stack(outs_list).unsqueeze(0)       
#         # doc_emb = out[:, :, 0, :] #take [cls] token embedding
#         # assert doc_emb.size() == (batch_size, chunk_size, self.hidden_size) # [batch_size, chunk_size, hidden_size]
#         # del text_id, outs_list, out
#         # gc.collect()
#         #-----------------------------------------------------------------------------------
#         text_id = text_id.view(-1, text_id.size(-1))
#         input_mask = ~(text_id == pad_id)
#         out = self.encoder(text_id, attention_mask=input_mask)[0]
#         doc_emb = out[:, 0, :].view(batch_size, chunk_size, self.hidden_size)  # [batch_size, doc_num, hidden_size]
#         assert doc_emb.size() == (batch_size, chunk_size, self.hidden_size)
#         del text_id, input_mask, out
#         gc.collect()        
#         #-----------------------------------------------------------------------------------

#         # Get summary embedding
#         input_mask = ~(summary_id == pad_id)
#         out = self.encoder(summary_id, attention_mask=input_mask)[0] # last layer
#         summary_emb = out[:, 0, :]
#         assert summary_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

#         # Get summary score
#         summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)
#         summary_score = summary_score.mean()
    
#         del input_mask, out, summary_emb
#         gc.collect()
#         #-----------------------------------------------------------------------------------

#         # Get candidate embedding
#         candidate_num = candidate_id.size(1)
#         candidate_id = candidate_id.view(-1, candidate_id.size(-1))
#         input_mask = ~(candidate_id == pad_id)
#         out = self.encoder(candidate_id, attention_mask=input_mask)[0]
#         candidate_emb = out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
#         assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)

#         del input_mask, out, candidate_id
#         gc.collect()        
#         #-----------------------------------------------------------------------------------

#         # get candidate score
#         scores_list = []
#         for i in range(chunk_size):              
#             chunk_doc_emb = doc_emb[:,i,:].unsqueeze(1).expand_as(candidate_emb)
#             sc = torch.cosine_similarity(candidate_emb, chunk_doc_emb, dim=-1) # [batch_size, candidate_num]
#             assert sc.size() == (batch_size, candidate_num)
#             scores_list.append(sc.squeeze(0)) 
#             del chunk_doc_emb, sc
#             gc.collect()

#         score_all = torch.stack(scores_list).unsqueeze(0)
#         score = torch.mean(score_all, dim=1)

#         return {'score': score, 'summary_score': summary_score}

