import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Local_ETHZ.local_ctx_att_ranker import LocalCtxAttRanker
from torch.distributions import Categorical
from Local_ETHZ.gcn.model import GCN
import Local_ETHZ.gcn.utils as gcnutil
import copy
import ipdb

np.set_printoptions(threshold=20)


class MulRelRanker(LocalCtxAttRanker):
    """
    multi-relational global model with context token attention, using loopy belief propagation
    """

    def __init__(self, config):

        print('--- create MulRelRanker model ---')
        super(MulRelRanker, self).__init__(config)
        self.dr = config['dr']
        self.gamma = config['gamma']
        # self.tok_top_n4ment = config['tok_top_n4ment']
        # self.tok_top_n4ent = config['tok_top_n4ent']
        # self.tok_top_n4word = config['tok_top_n4word']
        # self.tok_top_n4inlink = config['tok_top_n4inlink']
        # self.order_learning = config['order_learning']
        # self.dca_method = config['dca_method']

        self.ent_unk_id = config['entity_voca'].unk_id
        self.word_unk_id = config['word_voca'].unk_id
        # self.ent_inlinks = config['entity_inlinks']

        # self.oracle = config.get('oracle', False)
        self.use_local = config.get('use_local', False)
        # self.use_local_only = config.get('use_local_only', False)
        self.freeze_local = config.get('freeze_local', False)
        self.emb_dims = config['emb_dims']

        # self.entity2entity_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))
        # self.entity2entity_score_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))

        # self.knowledge2entity_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))
        # self.knowledge2entity_score_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))

        # self.ment2ment_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))
        # self.ment2ment_score_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))

        self.cnn = torch.nn.Conv1d(self.emb_dims, 64, kernel_size=3)
        self.gcn = GCN(self.emb_dims, self.emb_dims, self.emb_dims, config['gdr'])
        self.cnn_mgraph = torch.nn.Conv1d(self.emb_dims, self.emb_dims, kernel_size=5)
        self.m_e_score = torch.nn.Linear(2 * self.emb_dims, 1)

        self.saved_log_probs = []
        self.rewards = []
        self.actions = []

        self.order_saved_log_probs = []
        self.decision_order = []
        self.targets = []
        self.record = False
        if self.freeze_local:
            self.att_mat_diag.requires_grad = False
            self.tok_score_mat_diag.requires_grad = False

        # self.ment2ment_mat_diag.requires_grad = False
        # self.ment2ment_score_mat_diag.requires_grad = False
        self.param_copy_switch = True

        # Typing feature
        self.type_emb = torch.nn.Parameter(torch.randn([4, 5]))
        self.gcned_mat_diag = torch.nn.Parameter(torch.randn(self.emb_dims))
        self.score_combine = torch.nn.Sequential(
                torch.nn.Linear(5, self.hid_dims),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dr),
                torch.nn.Linear(self.hid_dims, 1))

        self.flag = 0
        self.doc_predict_restore = False
        # print('---------------- model config -----------------')
        # for k, v in self.__dict__.items():
        #     if not hasattr(v, '__dict__'):
        #         print(k, v)
        # print('-----------------------------------------------')

    def get_vec_of_graph(self, graph_embs, node_mask):
        # graph_embs: n_node * emb_dim
        return torch.mean(graph_embs, dim=0)

    def compute_gcned_similarity(self, entity_embs, ment_embs, isTrain=True):
        if isTrain:
            # entity_embs: n_ment * (n_sample+1) * n_node * emb_dim
            # ment_embs: n_ment * emb_dim
            # try to return: n_ment * (n_sample+1)
            n_ment, n_sample, n_node, _ = entity_embs.size()
            assert entity_embs.size(3) == self.emb_dims
            n_sample = n_sample - 1

            msk = torch.zeros(n_ment, n_node).cuda()
            msk[:n_ment, :n_ment] = torch.eye(n_ment)
            msk = msk.unsqueeze(1).unsqueeze(3).repeat(1, n_sample+1, 1, self.emb_dims)
            entity_embs2 = entity_embs.mul(msk).sum(dim=2)
            # entity_embs2: n_ment * (n_sample+1) * emb_dim
            sim_scores = torch.bmm(entity_embs2 * self.gcned_mat_diag, ment_embs.unsqueeze(2))
            # sim_scores: n_ment * (n_sample+1) * 1
            return sim_scores.squeeze(2)

        else:
            # entity_embs: search_ment_size * search_entity_size * n_node * emb_dim
            # ment_embs: n_ment * emb_dim
            # try to return: search_ment_size * search_entity_size * n_ment
            search_ment_size, search_entity_size, n_node, _ = entity_embs.size()
            n_ment = ment_embs.size(0)
            assert entity_embs.size(3) == self.emb_dims
            assert ment_embs.size(1) == self.emb_dims
            e_embs = entity_embs[:,:,:n_ment,:].reshape(-1, n_ment, self.emb_dims)
            # e_embs: (search_ment_size * search_entity_size) * n_ment * emb_dim
            sim_scores = (e_embs * self.gcned_mat_diag).mul(ment_embs).sum(dim=2)
            return sim_scores.view(search_ment_size, search_entity_size, n_ment)

    # def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, gold=None,
    #             method="SL", isTrain=True, isDynamic=0, isOrderLearning=False, isOrderFixed=False, isSort='topic', basemodel='deeped'):
    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, m_graph_list, m_graph_adj, nega_e, sample_idx, gold=None, method="SL", isTrain=True, chosen_ment=False):

        n_ments, n_cands = entity_ids.size()

        # cnn
        desc_len = desc_ids.size(-1)

        context_len = token_ids.size(-1)

        desc_ids = desc_ids.view(n_ments*n_cands, -1)
        desc_mask = desc_mask.view(n_ments*n_cands, 1, -1)

        if isTrain or (self.doc_predict_restore == False):
            context_emb = self.word_embeddings(token_ids)
            desc_emb = self.word_embeddings(desc_ids)

            # context_cnn: n_ment * 1 * 64
            context_cnn = F.max_pool1d(self.cnn(context_emb.permute(0, 2, 1)), context_len-2).permute(0, 2, 1)

            # desc_cnn: n_ment * n_cands * 64
            desc_cnn = F.max_pool1d(self.cnn(desc_emb.permute(0, 2, 1))-(1-desc_mask.float())*1e10, desc_len-2).view(n_ments, n_cands, -1)

            sim = torch.sum(context_cnn*desc_cnn,-1) / torch.sqrt(torch.sum(context_cnn*context_cnn, -1)) / torch.sqrt(torch.sum(desc_cnn*desc_cnn, -1))

            # if not self.oracle:
            #     gold = None

            # Typing feature
            self.mt_emb = torch.matmul(mtype, self.type_emb).view(n_ments, 1, -1)
            self.et_emb = torch.matmul(etype.view(-1, 4), self.type_emb).view(n_ments, n_cands, -1)
            tm = torch.sum(self.mt_emb*self.et_emb, -1, True)

            if self.use_local:
                local_ent_scores = super(MulRelRanker, self).forward(token_ids, tok_mask, entity_ids, entity_mask,p_e_m=None)
            else:
                local_ent_scores = Variable(torch.zeros(n_ments, n_cands).cuda(), requires_grad=False)
            # ment_emb: n_ment * emb_dim (only one graph)
            ment_emb = F.max_pool1d(self.cnn_mgraph(context_emb.permute(0, 2, 1)), context_len-4).squeeze(2)

            local_ent_scores = local_ent_scores.view(n_ments, n_cands)
            p_e_m = torch.log(p_e_m + 1e-20).view(n_ments, n_cands)
            tm = tm.view(n_ments, n_cands)

#             if not isTrain:
#                 self.doc_predict_restore = (local_ent_scores, tm, ment_emb)
        else:
            local_ent_scores, tm, ment_emb = self.doc_predict_restore

        nega_adjs, nega_node_cands, nega_node_mask = nega_e
        
        if isTrain or type(chosen_ment) != bool:

            # nega_adjs: n_ment * (n_sample+1) * n_node * n_node
            # nega_node_cands: n_ment * (n_sample+1) * n_node
            # nega_node_mask: n_ment * (n_sample+1) * n_node
            # nega_entity_emb: n_ment * (n_sample+1) * n_node * emb_dim
            nega_entity_emb = self.entity_embeddings(nega_node_cands)

            aaa, bbb, n_node, emb_dim = nega_entity_emb.size()
            ment_emb_r = gcnutil.feature_norm(ment_emb)
            nega_entity_emb_r = gcnutil.batch_feature_norm(nega_entity_emb.view(-1, n_node, emb_dim)).view(aaa, bbb, n_node, emb_dim)
            ment_emb_2 = self.gcn(ment_emb_r, m_graph_adj.long())
            nega_entity_emb_2 = self.gcn.batch_forward(nega_entity_emb_r.view(-1, n_node, emb_dim), nega_adjs.view(-1, n_node, n_node)).view(aaa, bbb, n_node, emb_dim)

            n_sample = nega_adjs.size(1) - 1

            # mention_graph_emb: emb_dim
            mention_graph_emb = torch.mean(ment_emb_2, dim=0)
            nega_node_mask2 = nega_node_mask.unsqueeze(3).repeat(1,1,1,self.emb_dims)
            nega_graph_embs = torch.sum(nega_entity_emb_2.mul(nega_node_mask2), dim=2)
            nega_node_mask2 = torch.sum(nega_node_mask2, dim=2)
            # nega_graph_embs = n_ment * (n_sample+1) * emb_dim
            # nega_entity_emb_2: n_ment * (n_sample+1) * n_node * emb_dim
            nega_graph_embs = torch.div(nega_graph_embs, nega_node_mask2)

            n_input = nega_graph_embs.size(0) * (n_sample+1)
            mention_graph_emb = mention_graph_emb.unsqueeze(0).repeat(n_input, 1)
            nega_graph_embs = nega_graph_embs.view(n_input, self.emb_dims)
            # graph_scores: (n_ment * (n_sample+1))
            graph_scores = self.m_e_score(torch.cat([mention_graph_emb, nega_graph_embs], dim=1))

            if isTrain:
                # gold: n_ment * 1
                # sample_idx: n_ment * n_sample
                sample_idx2 = torch.cat([sample_idx, gold], dim=1)
                # print("sample_idx2:", sample_idx2)

                sample_local_ent_scores = torch.gather(local_ent_scores, 1, sample_idx2)
                sample_p_e_m = torch.gather(p_e_m, 1, sample_idx2)
                sample_tm = torch.gather(tm, 1, sample_idx2)
                sample_gcnscore = self.compute_gcned_similarity(nega_entity_emb_2, ment_emb_2)
                    
                assert sample_local_ent_scores.size() == (n_ments, n_sample+1)
                assert sample_p_e_m.size() == (n_ments, n_sample+1)
                assert sample_tm.size() == (n_ments, n_sample+1)
                assert sample_gcnscore.size() == (n_ments, n_sample+1)

            else:
                # choosing the next-step graph entities while predicting
                #   n_ment == search_ment_size
                #   n_sample+1 == search_entity_size
                n_ments = nega_entity_emb_2.size(0)
                n_input = n_ments * (n_sample+1)

                # chosen_ment: LongTensor(search_ment_size)
                # sample_idx: search_ment_size * search_entity_size
                sample_local_ent_scores = torch.gather(local_ent_scores[chosen_ment], 1, sample_idx)
                sample_p_e_m = torch.gather(p_e_m[chosen_ment], 1, sample_idx)
                sample_tm = torch.gather(tm[chosen_ment], 1, sample_idx)
                sample_gcnscore = self.compute_gcned_similarity(nega_entity_emb_2, ment_emb_2, isTrain=False)[:,:,chosen_ment]
                # sample_gcnscore: search_ment_size * search_entity_size * search_ment_size
                msk = torch.eye(n_ments).cuda().unsqueeze(1).repeat(1,n_sample+1,1)
                # print("msk:", msk.size())
                # print("sample_gcnscore:", sample_gcnscore.size())
                sample_gcnscore = torch.sum(sample_gcnscore * msk, dim=2)

            #if self.use_local_only:
                # Typing feature
            inputs = torch.cat([sample_local_ent_scores.view(n_input, -1), sample_p_e_m.view(n_input, -1), sample_tm.view(n_input, -1), graph_scores.view(n_input, -1), sample_gcnscore.view(n_input, -1)], dim=1)

            scores = self.score_combine(inputs).view(n_ments, n_sample+1)

        else:
            # calculating scores of all current ment-entity pairs
            # to choose some entities to be replaced
            # nega_adjs: n_node * n_node
            # nega_node_cands: n_node
            # nega_node_mask: n_node
            # nega_entity_emb: n_node * emb_dim
            nega_entity_emb = self.entity_embeddings(nega_node_cands)
            ment_emb_r = gcnutil.feature_norm(ment_emb)
            nega_entity_emb_r = gcnutil.feature_norm(nega_entity_emb)
            ment_emb_2 = self.gcn(ment_emb_r, m_graph_adj)
            nega_entity_emb_2 = self.gcn(nega_entity_emb_r, nega_adjs)

            mention_graph_emb = torch.mean(ment_emb_2, dim=0)
            entity_graph_emb = torch.mean(nega_entity_emb_2, dim=0)
            graph_scores = self.m_e_score(torch.cat([mention_graph_emb, entity_graph_emb], dim=0).unsqueeze(0))
            graph_scores = graph_scores * torch.ones(n_ments, 1).cuda()

            sample_idx2 = sample_idx.unsqueeze(1)

            sample_local_ent_scores = torch.gather(local_ent_scores, 1, sample_idx2)
            sample_p_e_m = torch.gather(p_e_m, 1, sample_idx2)
            sample_tm = torch.gather(tm, 1, sample_idx2)

            entity_emb = nega_entity_emb[:n_ments]
            # entity_emb: n_ment * emb_dims
            # ment_emb: n_ment * emb_dims
            sample_gcnscore = torch.mul(ment_emb*self.gcned_mat_diag, entity_emb).sum(dim=1).unsqueeze(1)

            n_input = n_ments
            inputs = torch.cat([sample_local_ent_scores, sample_p_e_m, sample_tm, graph_scores, sample_gcnscore], dim=1)
            scores = self.score_combine(inputs)

        # inputs = torch.cat([local_ent_scores.view(n_ments * n_cands, -1),
        #                     torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1)], dim=1)
        # print("n_ments, n_cands", n_ments, n_cands)
        # print("desc_len, context_len", desc_len, context_len)
        # print("desc_ids", desc_ids.size())
        # print("desc_mask", desc_mask.size())
        # print("input",inputs.size())
        # print("local_ent_scores",local_ent_scores.size())
        # print("p_e_m",p_e_m.size())
        # print("tm",tm.size())
        # print("self.score_combine",self.score_combine)
        
        # assert False
        
        if torch.isnan(scores).any():
            ipdb.set_trace()
        return scores, self.actions

    def unique(self, numpy_array):
        t = np.unique(numpy_array)
        return torch.from_numpy(t).type(torch.LongTensor)

    ###############################  DELETING #################################

    # def compute_coherence(self, cumulative_ids, entity_ids, entity_mask, att_mat_diag, score_att_mat_diag, window_size, isWord=False):
    #     n_cumulative_entities = cumulative_ids.size(0)
    #     n_entities = entity_ids.size(0)

    #     if self.dca_method == 1 or self.dca_method == 2:
    #         window_size = 100

    #     print('n_cumulative_entities', n_cumulative_entities)
    #     print('n_entities', n_entities)

    #     try:
    #         if isWord:
    #             cumulative_entity_vecs = self.word_embeddings(cumulative_ids)
    #         else:
    #             cumulative_entity_vecs = self.entity_embeddings(cumulative_ids)
    #     except:
    #         print(cumulative_ids)
    #         input()

    #     # cumulative_entity_vecs = self.entity_embeddings(cumulative_ids)

    #     entity_vecs = self.entity_embeddings(entity_ids)

    #     # debug
    #     # print("Cumulative_entity_ids Size: ", cumulative_ids.size(), cumulative_ids.size(0))
    #     # print("Entity_ids Size: ", entity_ids.size(), entity_ids.size(0))
    #     # print("Cumulative_entity_vecs Size: ", cumulative_entity_vecs.size())
    #     # print("Entity_vecs Size: ", entity_vecs.size())

    #     # att
    #     ent2ent_att_scores = torch.mm(entity_vecs * att_mat_diag, cumulative_entity_vecs.permute(1, 0))
    #     ent_tok_att_scores, _ = torch.max(ent2ent_att_scores, dim=0)
    #     top_tok_att_scores, top_tok_att_ids = torch.topk(ent_tok_att_scores, dim=0, k=min(window_size, n_cumulative_entities))

    #     # print("Top_tok_att_scores Size: ", top_tok_att_scores.size())
    #     # print("Top_tok_att_scores: ", top_tok_att_scores)
    #     if self.dca_method == 2:
    #         entity_att_probs = F.softmax(top_tok_att_scores*0., dim=0).view(-1, 1)
    #     else:
    #         entity_att_probs = F.softmax(top_tok_att_scores, dim=0).view(-1, 1)
    #     entity_att_probs = entity_att_probs / torch.sum(entity_att_probs, dim=0, keepdim=True)

    #     # print("entity_att_probs: ", entity_att_probs)

    #     selected_tok_vecs = torch.gather(cumulative_entity_vecs, dim=0,
    #                                      index=top_tok_att_ids.view(-1, 1).repeat(1, cumulative_entity_vecs.size(1)))

    #     ctx_ent_vecs = torch.sum((selected_tok_vecs * score_att_mat_diag) * entity_att_probs, dim=0, keepdim=True)

    #     # print("Selected_vecs * diag Size: ", (selected_tok_vecs * att_mat_diag).size())
    #     # print("Before Sum Size: ", ((selected_tok_vecs * att_mat_diag) * entity_att_probs).size())
    #     # print("Ctx_ent_vecs Size: ", ctx_ent_vecs.size())

    #     ent_ctx_scores = torch.mm(entity_vecs, ctx_ent_vecs.permute(1, 0)).view(-1, n_entities)

    #     # print("Ent_ctx_scores", ent_ctx_scores)

    #     scores = (ent_ctx_scores * entity_mask).add_((entity_mask - 1).mul_(1e10))

    #     # print("Scores: ", scores)

    #     return scores, cumulative_ids[top_tok_att_ids.view(-1)].view(-1)

    ##########################  DELETING #####################################

    def print_weight_norm(self):
        LocalCtxAttRanker.print_weight_norm(self)

        # print('entity2entity_mat_diag', self.entity2entity_mat_diag.data.norm())
        # print('entity2entity_score_mat_diag', self.entity2entity_score_mat_diag.data.norm())

        # print('knowledge2entity_mat_diag', self.knowledge2entity_mat_diag.data.norm())
        # print('knowledge2entity_score_mat_diag', self.knowledge2entity_score_mat_diag.data.norm())

        # print('ment2ment_mat_diag', self.ment2ment_mat_diag.data.norm())
        # print('ment2ment_score_mat_diag', self.ment2ment_score_mat_diag.data.norm())

        print('f - l1.w, b', self.score_combine[0].weight.data.norm(), self.score_combine[0].bias.data.norm())
        print('f - l2.w, b', self.score_combine[3].weight.data.norm(), self.score_combine[3].bias.data.norm())

        # print(self.ctx_layer[0].weight.data.norm(), self.ctx_layer[0].bias.data.norm())
        # print('relations', self.rel_embs.data.norm(p=2, dim=1))
        # X = F.normalize(self.rel_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).sqrt()
        # print(diff)
        #
        # print('ew_embs', self.ew_embs.data.norm(p=2, dim=1))
        # X = F.normalize(self.ew_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).sqrt()
        # print(diff)

    def regularize(self, max_norm=4):
        # super(MulRelRanker, self).regularize(max_norm)
        # print("----MulRelRanker Regularization----")

        l1_w_norm = self.score_combine[0].weight.norm()
        l1_b_norm = self.score_combine[0].bias.norm()
        l2_w_norm = self.score_combine[3].weight.norm()
        l2_b_norm = self.score_combine[3].bias.norm()

        if (l1_w_norm > max_norm).data.all():
            self.score_combine[0].weight.data = self.score_combine[0].weight.data * max_norm / l1_w_norm.data
        if (l1_b_norm > max_norm).data.all():
            self.score_combine[0].bias.data = self.score_combine[0].bias.data * max_norm / l1_b_norm.data
        if (l2_w_norm > max_norm).data.all():
            self.score_combine[3].weight.data = self.score_combine[3].weight.data * max_norm / l2_w_norm.data
        if (l2_b_norm > max_norm).data.all():
            self.score_combine[3].bias.data = self.score_combine[3].bias.data * max_norm / l2_b_norm.data

    def finish_episode(self, rewards_arr, log_prob_arr):
        if len(rewards_arr) != len(log_prob_arr):
            print("Size mismatch between Rwards and Log_probs!")
            return

        policy_loss = []
        rewards = []

        # we only give a non-zero reward when done
        g_return = sum(rewards_arr) / len(rewards_arr)

        # add the final return in the last step
        rewards.insert(0, g_return)

        R = g_return
        for idx in range(len(rewards_arr) - 1):
            R = R * self.gamma
            rewards.insert(0, R)

        rewards = torch.from_numpy(np.array(rewards)).type(torch.cuda.FloatTensor)

        for log_prob, reward in zip(log_prob_arr, rewards):
            policy_loss.append(-log_prob * reward)

        policy_loss = torch.cat(policy_loss).sum()

        return policy_loss

    def loss(self, scores, true_pos, method="SL", lamb=1e-7):
        loss = None

        # print("----MulRelRanker Loss----")
        if method == "SL":
            loss = F.multi_margin_loss(scores, true_pos, margin=self.margin)
        elif method == "RL":
            loss = self.finish_episode(self.rewards, self.saved_log_probs)

        return loss

    def order_loss(self):
        return self.finish_episode(self.rewards[1:], self.order_saved_log_probs)

    def get_order_truth(self):
        return self.decision_order, self.targets
