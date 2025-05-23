from collections import OrderedDict
import torch
import torch.nn as nn
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import mat_heatmap
import psd2.utils.comm as comm

#TODO weight decay compatibility
class CodaPrompt(nn.Module):
    def __init__(
        self,
        emb_d,
        n_tasks,
        pool_size, 
        num_prompts,
        num_layers,
        loss_weight,
        vis_period,
        key_dim=768,
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(pool_size, num_prompts, num_layers, loss_weight)
        self.vis_period = vis_period

        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d, ortho=True)
            k = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
            a = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)
            setattr(self, f"e_a_{e}", a)

    def _init_smart(self, pool_size, num_prompts, num_layers, ortho_mu):
        # prompt basic param
        self.e_pool_size = int(pool_size)
        self.e_p_length = int(num_prompts)
        self.e_layers = list(range(num_layers))

        # strenth of ortho penalty
        self.ortho_mu = ortho_mu

    def process_task_count(self, task_id):
        self.task_count = task_id

    def forward(self, x_query, vis_mark, train=False, task_id=None,domain_id=False):
        """
        select all layers in one pass
        NOTE assume x_query to be batch_size * num_layer * c
        """
        B, nL, C = x_query.shape
        assert nL == len(self.e_layers)
        # e prompts
        p_return = []
        p_loss = 0
        vis_attn = []
        for l in range(nL):
            K = getattr(self, f"e_k_{l}")
            A = getattr(self, f"e_a_{l}")
            p = getattr(self, f"e_p_{l}")
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_query = torch.einsum("bd,kd->bkd", x_query[:, l, :], A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_query, dim=2)
            aq_k = torch.einsum("bkd,kd->bk", q, n_K)
            if train:
                vis_attn.append(aq_k.detach())
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum("bk,kld->bld", aq_k, p)
            p_return.append(P_)

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K)
                loss += ortho_penalty(A)
                loss += ortho_penalty(p.flatten(start_dim=1, end_dim=2))
                p_loss += loss * self.ortho_mu
        p_return = torch.stack(p_return, dim=0)
        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        if domain_id:
            d_attn=vis_attn.reshape(B,self.n_tasks,-1).mean(2)
            dids=torch.argmax(d_attn,dim=1)
            return dids, p_return, p_loss
        return p_return, p_loss


class CodaPromptWd(nn.Module):
    def __init__(
        self,
        emb_d,
        n_tasks,
        pool_size,
        num_prompts,
        num_layers,
        loss_weight,
        vis_period,
        key_dim=768,
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(pool_size, num_prompts, num_layers, loss_weight)
        self.vis_period = vis_period
        pool_size_task = pool_size // n_tasks
        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt_value(self.e_pool_size, e_l, emb_d, ortho=True)
            k = tensor_prompt_value(self.e_pool_size, self.key_d, ortho=True)
            a = tensor_prompt_value(self.e_pool_size, self.key_d, ortho=True)
            for ti in range(n_tasks):
                s = ti * pool_size_task
                f = (ti + 1) * pool_size_task
                if ti == 0:
                    setattr(
                        self,
                        f"e_p_l{e}_t{ti}",
                        torch.nn.Parameter(p[s:f], requires_grad=True),
                    )
                    setattr(
                        self,
                        f"e_k_l{e}_t{ti}",
                        torch.nn.Parameter(k[s:f], requires_grad=True),
                    )
                    setattr(
                        self,
                        f"e_a_l{e}_t{ti}",
                        torch.nn.Parameter(a[s:f], requires_grad=True),
                    )
                else:
                    setattr(
                        self,
                        f"e_p_l{e}_t{ti}",
                        torch.nn.Parameter(p[s:f], requires_grad=False),
                    )
                    setattr(
                        self,
                        f"e_k_l{e}_t{ti}",
                        torch.nn.Parameter(k[s:f], requires_grad=False),
                    )
                    setattr(
                        self,
                        f"e_a_l{e}_t{ti}",
                        torch.nn.Parameter(a[s:f], requires_grad=False),
                    )

    def _init_smart(self, pool_size, num_prompts, num_layers, ortho_mu):
        # prompt basic param
        self.e_pool_size = int(pool_size)
        self.e_p_length = int(num_prompts)
        self.e_layers = list(range(num_layers))

        # strenth of ortho penalty
        self.ortho_mu = ortho_mu

    def process_task_count(self, task_id):
        self.task_count = task_id
        for e in self.e_layers:
            for ti in range(self.n_tasks):
                if ti != task_id:
                    getattr(self, f"e_p_l{e}_t{ti}").requires_grad = False
                    getattr(self, f"e_k_l{e}_t{ti}").requires_grad = False
                    getattr(self, f"e_a_l{e}_t{ti}").requires_grad = False
                else:
                    getattr(self, f"e_p_l{e}_t{ti}").requires_grad = True
                    getattr(self, f"e_k_l{e}_t{ti}").requires_grad = True
                    getattr(self, f"e_a_l{e}_t{ti}").requires_grad = True

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        """
        select all layers in one pass
        NOTE assume x_query to be batch_size * num_layer * c
        """
        B, nL, C = x_query.shape
        assert nL == len(self.e_layers)
        # e prompts
        p_return = []
        p_loss = 0
        vis_attn = []
        for l in range(nL):
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat(
                        [
                            getattr(self, f"e_k_l{l}_t{ti}").clone()
                            for ti in range(self.task_count)
                        ]
                        + [getattr(self, f"e_k_l{l}_t{self.task_count}")],
                        dim=0,
                    )
                    A = torch.cat(
                        [
                            getattr(self, f"e_a_l{l}_t{ti}").clone()
                            for ti in range(self.task_count)
                        ]
                        + [getattr(self, f"e_a_l{l}_t{self.task_count}")],
                        dim=0,
                    )
                    p = torch.cat(
                        [
                            getattr(self, f"e_p_l{l}_t{ti}").clone()
                            for ti in range(self.task_count)
                        ]
                        + [getattr(self, f"e_p_l{l}_t{self.task_count}")],
                        dim=0,
                    )
                else:
                    K = getattr(self, f"e_k_l{l}_t0")
                    A = getattr(self, f"e_a_l{l}_t0")
                    p = getattr(self, f"e_p_l{l}_t0")
            else:
                K = torch.cat(
                    [
                        getattr(self, f"e_k_l{l}_t{ti}")
                        for ti in range(self.task_count + 1)
                    ],
                    dim=0,
                )
                A = torch.cat(
                    [
                        getattr(self, f"e_a_l{l}_t{ti}")
                        for ti in range(self.task_count + 1)
                    ],
                    dim=0,
                )
                p = torch.cat(
                    [
                        getattr(self, f"e_p_l{l}_t{ti}")
                        for ti in range(self.task_count + 1)
                    ],
                    dim=0,
                )

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_query = torch.einsum("bd,kd->bkd", x_query[:, l, :], A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_query, dim=2)
            aq_k = torch.einsum("bkd,kd->bk", q, n_K)
            if train:
                vis_attn.append(aq_k.detach())
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum("bk,kld->bld", aq_k, p)
            p_return.append(P_)

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K)
                loss += ortho_penalty(A)
                loss += ortho_penalty(p.flatten(start_dim=1, end_dim=2))
                p_loss += loss * self.ortho_mu
        p_return = torch.stack(p_return, dim=0)
        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        return p_return, p_loss


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean() * 1e-6 #NOTE

def dis_sim_penalty(t1,t2):
    sim=torch.mm(t1,t2.T)
    return sim.mean()

def dis_orth_penalty(t1,t2):
    return ((t1 @ t2.T ) ** 2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(
        self,
        emb_d,
        n_tasks,
        e_pool_size,
        num_e_prompts,
        num_g_prompts,
        e_layers,
        g_layers,
        loss_weight,
        topk,
        vis_period,
        key_dim=768,
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.top_k = topk
        # n_tasks==e_pool_size in dual-p
        self.loss_weight = loss_weight
        self._init_smart(e_pool_size, num_e_prompts, num_g_prompts, e_layers, g_layers)
        self.vis_period = vis_period
        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f"g_p_{g}", p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)

    def _init_smart(
        self, e_pool_size, num_e_prompts, num_g_prompts, e_layers, g_layers
    ):
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = g_layers
        self.e_layers = e_layers

        # prompt pool size
        self.g_p_length = num_g_prompts
        self.e_p_length = num_e_prompts
        self.e_pool_size = e_pool_size

    def process_task_count(self, task_id):
        self.task_count = task_id

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        B, nL, C = x_query.shape
        p_return = []
        p_loss = 0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here

                # cosine similarity to match keys/querries
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_query[:, l, :], dim=1).detach()
                cos_sim = torch.einsum("bj,kj->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())
                    # dual prompt during training uses task id
                    if self.task_id_bootstrap:
                        loss = (
                            1.0
                            - cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ]
                        ).sum()
                        P_ = (
                            p[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k
                            ]
                            .flatten(0, 1)
                            .unsqueeze(0)
                            .expand(B, -1, -1)
                        )  # B, num_e_prompts*topk, d
                    else:
                        top_k = torch.topk(cos_sim, self.top_k, dim=1)
                        k_idx = top_k.indices
                        loss = (1.0 - cos_sim[:, k_idx]).sum()
                        P_ = p[k_idx]
                    p_loss += loss
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    P_ = p[k_idx]
                lp.append(P_)
            # g prompts
            if l in self.g_layers:
                p = getattr(self, f"g_p_{l}")  # 0 based indexing here
                P_ = p.expand(B, -1, -1)
                lp.append(P_)
            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train and len(vis_attn)>0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        
        return p_return, p_loss * self.loss_weight


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2Ppp(DualPrompt):
    def __init__(
        self,
        emb_d,
        n_tasks,
        pool_size,
        num_prompts,
        num_layers,
        loss_weight,
        topk,
        vis_period,
        key_dim=768,
    ):
        super().__init__(
            emb_d,
            n_tasks,
            pool_size,
            num_prompts,
            -1,
            list(range(num_layers)),
            [],
            loss_weight,
            topk,
            vis_period,
            key_dim,
        )

    def _init_smart(
        self, e_pool_size, num_e_prompts, num_g_prompts, e_layers, g_layers
    ):
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = g_layers
        self.e_layers = e_layers

        # prompt pool size
        self.g_p_length = num_g_prompts
        self.e_p_length = num_e_prompts
        self.e_pool_size = e_pool_size

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        p_list, ploss = super().forward(x_query, vis_mark, train, task_id)
        return torch.stack(p_list, dim=0).flatten(2, 3), ploss


# NOTE use prompt mask
class L2PppMask(L2Ppp):
    def _init_smart(self, *args, **kws):
        super()._init_smart(*args, **kws)
        self.task_id_bootstrap = True

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        if self.training:
            p_list, ploss = super(L2Ppp, self).forward(
                x_query, vis_mark, train, task_id
            )
            return torch.stack(p_list, dim=0), ploss
        else:
            return super().forward(x_query, vis_mark, train, task_id)

class L2PppMaskOrth(L2Ppp):
    def __init__(
        self,orth_mu,*args, **kws):
        super().__init__(*args, **kws)
        self.orth_mu=orth_mu
    def _init_smart(self, *args, **kws):
        super()._init_smart(*args, **kws)
        self.task_id_bootstrap = True

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        B, nL, C = x_query.shape
        p_return = []
        p_loss = 0.0
        orth_loss=0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here

                # cosine similarity to match keys/querries
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_query[:, l, :], dim=1).detach()
                cos_sim = torch.einsum("bj,kj->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())
                    # dual prompt during training uses task id
                    loss = (
                            1.0
                            - cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ]
                        ).sum()
                    n_K_prev=n_K[0:self.task_count*self.top_k,:].detach()
                    n_K_cur=n_K[self.task_count*self.top_k:(self.task_count + 1)
                                * self.top_k,:]
                    o_loss=ortho_penalty(torch.cat([n_K_prev,n_K_cur],dim=0))
                    P_ = (
                            p[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k
                            ]
                            .flatten(0, 1)
                            .unsqueeze(0)
                            .expand(B, -1, -1)
                        )  # B, num_e_prompts*topk, d
                    p_loss += loss
                    orth_loss+=o_loss
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    P_ = p[k_idx]
                lp.append(P_)

            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        p_loss =p_loss* self.loss_weight
        orth_loss=orth_loss*self.orth_mu
        if self.training:
            return torch.stack(p_return, dim=0), [p_loss,orth_loss]
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), [p_loss,orth_loss]

class L2PppMaskSeOrth(L2Ppp):
    def __init__(
        self,orth_mu,*args, **kws):
        super().__init__(*args, **kws)
        self.orth_mu=orth_mu
        for e in self.e_layers:
            a = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f"e_a_{e}", a)
    def _init_smart(self, *args, **kws):
        super()._init_smart(*args, **kws)
        self.task_id_bootstrap = True

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        B, nL, C = x_query.shape
        p_return = []
        p_loss = 0.0
        orth_loss=0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here
                A = getattr(self, f"e_a_{l}")  # 0 based indexing here
                n_A=nn.functional.normalize(A, dim=1)

                # cosine similarity to match keys/querries
                a_query = torch.einsum("bd,kd->bkd", x_query[:, l, :], A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_query, dim=2)
                cos_sim = torch.einsum("bkd,kd->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())
                    # dual prompt during training uses task id
                    loss = (
                            1.0
                            - cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ]
                        ).sum()
                    o_loss=ortho_penalty(n_K[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])+ortho_penalty(n_A[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])
                    P_ = (
                            p[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k
                            ]
                            .flatten(0, 1)
                            .unsqueeze(0)
                            .expand(B, -1, -1)
                        )  # B, num_e_prompts*topk, d
                    p_loss += loss
                    orth_loss+=o_loss
                else:
                    per_task_sim=cos_sim.reshape(B,-1,self.top_k).sum(-1).sum(0)
                    task_id=torch.argmax(per_task_sim).item()
                    k_idx=torch.arange(task_id*self.top_k,(task_id+1)*self.top_k).unsqueeze(0).repeat(B,1)
                    P_ = p[k_idx]
                lp.append(P_)

            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        p_loss =p_loss* self.loss_weight
        orth_loss=orth_loss*self.orth_mu
        if self.training:
            return torch.stack(p_return, dim=0), [p_loss,orth_loss]
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), [p_loss,orth_loss]

class L2PppMaskSeOrthWD(nn.Module):
    def __init__(
        self,
        emb_d,
        n_tasks,
        pool_size,
        num_prompts,
        num_layers,
        loss_weight,
        topk,
        vis_period,
        orth_mu,
        key_dim=768,
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.top_k = topk
        self.loss_weight = loss_weight
        e_layers=list(range(num_layers))
        self._init_smart(pool_size, num_prompts,  e_layers)
        self.vis_period = vis_period
        self.orth_mu=orth_mu

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            for ti in range(n_tasks):
                s = ti * topk
                f = (ti + 1) * topk
                if ti == 0:
                    setattr(
                        self,
                        f"e_p_l{e}_t{ti}",
                        torch.nn.Parameter(p[s:f], requires_grad=True),
                    )
                    setattr(
                        self,
                        f"e_k_l{e}_t{ti}",
                        torch.nn.Parameter(k[s:f], requires_grad=True),
                    )
                    setattr(
                        self,
                        f"e_a_l{e}_t{ti}",
                        torch.nn.Parameter(a[s:f], requires_grad=True),
                    )
                else:
                    setattr(
                        self,
                        f"e_p_l{e}_t{ti}",
                        torch.nn.Parameter(p[s:f], requires_grad=False),
                    )
                    setattr(
                        self,
                        f"e_k_l{e}_t{ti}",
                        torch.nn.Parameter(k[s:f], requires_grad=False),
                    )
                    setattr(
                        self,
                        f"e_a_l{e}_t{ti}",
                        torch.nn.Parameter(a[s:f], requires_grad=False),
                    )

    def _init_smart(
        self, e_pool_size, num_e_prompts,  e_layers
    ):
        self.task_id_bootstrap = True

        # prompt locations
        self.e_layers = e_layers

        # prompt pool size
        self.e_p_length = num_e_prompts
        self.e_pool_size = e_pool_size

    def process_task_count(self, task_id):
        self.task_count = task_id
        for e in self.e_layers:
            for ti in range(self.n_tasks):
                if ti != task_id:
                    getattr(self, f"e_p_l{e}_t{ti}").requires_grad = False
                    getattr(self, f"e_k_l{e}_t{ti}").requires_grad = False
                    getattr(self, f"e_a_l{e}_t{ti}").requires_grad = False
                else:
                    getattr(self, f"e_p_l{e}_t{ti}").requires_grad = True
                    getattr(self, f"e_k_l{e}_t{ti}").requires_grad = True
                    getattr(self, f"e_a_l{e}_t{ti}").requires_grad = True
    def forward(self, x_query, vis_mark, train=False, task_id=None):
        B, nL, C = x_query.shape
        p_return = []
        p_loss = 0.0
        orth_loss=0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_l{l}_t{self.task_count}")  # 0 based indexing here
                p = getattr(self, f"e_p_l{l}_t{self.task_count}")  # 0 based indexing here
                A = getattr(self, f"e_a_l{l}_t{self.task_count}")  # 0 based indexing here
                n_A=nn.functional.normalize(A, dim=1)

                # cosine similarity to match keys/querries
                a_query = torch.einsum("bd,kd->bkd", x_query[:, l, :], A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_query, dim=2)
                cos_sim = torch.einsum("bkd,kd->bk", q, n_K)

                K_all = torch.cat(
                    [
                        getattr(self, f"e_k_l{l}_t{ti}")
                        for ti in range(self.n_tasks)
                    ],
                    dim=0,
                )
                A_all = torch.cat(
                    [
                        getattr(self, f"e_a_l{l}_t{ti}")
                        for ti in range(self.n_tasks)
                    ],
                    dim=0,
                )
                p_all = torch.cat(
                    [
                        getattr(self, f"e_p_l{l}_t{ti}")
                        for ti in range(self.n_tasks)
                    ],
                    dim=0,
                )

                a_query_all = torch.einsum("bd,kd->bkd", x_query[:, l, :], A_all)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K_all = nn.functional.normalize(K_all, dim=1)
                q_all = nn.functional.normalize(a_query_all, dim=2)
                cos_sim_all = torch.einsum("bkd,kd->bk", q_all, n_K_all)

                if train:
                    vis_attn.append(cos_sim_all.detach())
                    # dual prompt during training uses task id
                    loss = (
                            1.0
                            - cos_sim
                        ).sum()
                    o_loss=ortho_penalty(n_K)+ortho_penalty(n_A)
                    P_ = (
                            p.flatten(0, 1)
                            .unsqueeze(0)
                            .expand(B, -1, -1)
                        )  # B, num_e_prompts*topk, d
                    p_loss += loss
                    orth_loss+=o_loss
                else:
                    per_task_sim=cos_sim_all.reshape(B,-1,self.top_k).sum(-1).sum(0)
                    task_id=torch.argmax(per_task_sim).item()
                    k_idx=torch.arange(task_id*self.top_k,(task_id+1)*self.top_k).unsqueeze(0).repeat(B,1)
                    P_ = p_all[k_idx]
                lp.append(P_)

            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        p_loss =p_loss* self.loss_weight
        orth_loss=orth_loss*self.orth_mu
        if self.training:
            return torch.stack(p_return, dim=0), [p_loss,orth_loss]
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), [p_loss,orth_loss]


class L2PppMaskSeOrthTD(L2PppMaskSeOrth):
    def __init__(
        self,td_mu,*args, **kws):
        super().__init__(*args, **kws)
        self.td_mu=td_mu
    def forward(self, x_query, vis_mark, train=False, task_id=None):
        B, nL, C = x_query.shape
        p_return = []
        p_loss = 0.0
        orth_loss=0.0
        td_loss=0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here
                A = getattr(self, f"e_a_{l}")  # 0 based indexing here
                n_A=nn.functional.normalize(A, dim=1)

                # cosine similarity to match keys/querries
                a_query = torch.einsum("bd,kd->bkd", x_query[:, l, :], A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_query, dim=2)
                cos_sim = torch.einsum("bkd,kd->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())
                    # dual prompt during training uses task id
                    loss = (
                            1.0
                            - cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ]
                        ).sum()
                    o_loss=ortho_penalty(n_K[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])+ortho_penalty(n_A[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])
                    if self.task_count>0:
                        dis_loss=dis_sim_penalty(n_K[
                                0 : self.task_count 
                                * self.top_k,:],n_K[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])+dis_sim_penalty(n_A[
                                0 : self.task_count 
                                * self.top_k,:],n_A[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])
                        td_loss+=dis_loss
                    P_ = (
                            p[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k
                            ]
                            .flatten(0, 1)
                            .unsqueeze(0)
                            .expand(B, -1, -1)
                        )  # B, num_e_prompts*topk, d
                    p_loss += loss
                    orth_loss+=o_loss
                else:
                    per_task_sim=cos_sim.reshape(B,-1,self.top_k).sum(-1).sum(0)
                    task_id=torch.argmax(per_task_sim).item()
                    k_idx=torch.arange(task_id*self.top_k,(task_id+1)*self.top_k).unsqueeze(0).repeat(B,1)
                    P_ = p[k_idx]
                lp.append(P_)

            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        p_loss =p_loss* self.loss_weight
        orth_loss=orth_loss*self.orth_mu+td_loss*self.td_mu
        if self.training:
            return torch.stack(p_return, dim=0), [p_loss,orth_loss]
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), [p_loss,orth_loss]

class L2PppMaskSeOrthTO(L2PppMaskSeOrth):
    def __init__(
        self,to_mu,*args, **kws):
        super().__init__(*args, **kws)
        self.to_mu=to_mu
    def forward(self, x_query, vis_mark, train=False, task_id=None):
        B, nL, C = x_query.shape
        p_return = []
        p_loss = 0.0
        orth_loss=0.0
        td_loss=0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here
                A = getattr(self, f"e_a_{l}")  # 0 based indexing here
                n_A=nn.functional.normalize(A, dim=1)

                # cosine similarity to match keys/querries
                a_query = torch.einsum("bd,kd->bkd", x_query[:, l, :], A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_query, dim=2)
                cos_sim = torch.einsum("bkd,kd->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())
                    # dual prompt during training uses task id
                    loss = (
                            1.0
                            - cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ]
                        ).sum()
                    o_loss=ortho_penalty(n_K[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])+ortho_penalty(n_A[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])
                    if self.task_count>0:
                        dis_loss=dis_orth_penalty(n_K[
                                0 : self.task_count 
                                * self.top_k,:],n_K[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])+dis_orth_penalty(n_A[
                                0 : self.task_count 
                                * self.top_k,:],n_A[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])
                        td_loss+=dis_loss
                    P_ = (
                            p[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k
                            ]
                            .flatten(0, 1)
                            .unsqueeze(0)
                            .expand(B, -1, -1)
                        )  # B, num_e_prompts*topk, d
                    p_loss += loss
                    orth_loss+=o_loss
                else:
                    per_task_sim=cos_sim.reshape(B,-1,self.top_k).sum(-1).sum(0)
                    task_id=torch.argmax(per_task_sim).item()
                    k_idx=torch.arange(task_id*self.top_k,(task_id+1)*self.top_k).unsqueeze(0).repeat(B,1)
                    P_ = p[k_idx]
                lp.append(P_)

            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        p_loss =p_loss* self.loss_weight
        orth_loss=orth_loss*self.orth_mu+td_loss*self.to_mu
        if self.training:
            return torch.stack(p_return, dim=0), [p_loss,orth_loss]
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), [p_loss,orth_loss]

class L2PppMaskBs(L2Ppp):
    def _init_smart(self, *args, **kws):
        super()._init_smart(*args, **kws)
        self.task_id_bootstrap = True

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        if self.training:
            p_list, ploss = super(L2Ppp, self).forward(
                x_query, vis_mark, train, task_id
            )
            return torch.stack(p_list, dim=0), ploss/x_query.shape[0]
        else:
            return super().forward(x_query, vis_mark, train, task_id)


# NOTE use prompt mask
class L2PppMask2(L2Ppp):
    def _init_smart(self, *args, **kws):
        super()._init_smart(*args, **kws)
        self.loss_weight_neg = self.loss_weight[1]
        self.loss_weight = self.loss_weight[0]

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        B, nL, C = x_query.shape
        p_return = []
        p_loss = 0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here

                # cosine similarity to match keys/querries
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_query[:, l, :], dim=1).detach()
                cos_sim = torch.einsum("bj,kj->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())
                    loss = (
                        1.0
                        - cos_sim[
                            :,
                            self.task_count
                            * self.top_k : (self.task_count + 1)
                            * self.top_k,
                        ]
                    ).sum()
                    P_ = (
                        p[
                            self.task_count
                            * self.top_k : (self.task_count + 1)
                            * self.top_k
                        ]
                        .flatten(0, 1)
                        .unsqueeze(0)
                        .expand(B, -1, -1)
                    )  # B, num_e_prompts*topk, d
                    p_loss += loss * self.loss_weight
                    if self.task_count > 0:
                        # negtive loss
                        prev_keys_n = nn.functional.normalize(
                            K[: self.task_count * self.top_k].clone().detach(), dim=1
                        )
                        cur_keys_n = nn.functional.normalize(
                            K[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k
                            ],
                            dim=1,
                        )
                        neg_cos_sim = torch.einsum("bj,kj->bk", cur_keys_n, prev_keys_n)
                        p_loss += neg_cos_sim.sum() * self.loss_weight_neg
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    P_ = p[k_idx]
                lp.append(P_)
            # g prompts
            if l in self.g_layers:
                p = getattr(self, f"g_p_{l}")  # 0 based indexing here
                P_ = p.expand(B, -1, -1)
                lp.append(P_)
            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        if self.training:
            return torch.stack(p_return, dim=0), p_loss
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), p_loss


class L2PppMaskM(L2Ppp):
    def _init_smart(self, *args, **kws):
        super()._init_smart(*args, **kws)
        self.loss_weight_mem_neg = self.loss_weight[2]  # memory to new key
        self.loss_weight_mem_pos = self.loss_weight[3]  # memory to old key
        self.loss_weight_neg = self.loss_weight[1]  # data to old key
        self.loss_weight = self.loss_weight[0]  # data to new key
        self.mem_len = 512
        self.mem_batch = 64
        self.register_buffer(
            "memory", torch.zeros(self.n_tasks, self.mem_len, self.key_d)
        )
        self.register_buffer(
            "memory_next", torch.zeros(self.n_tasks, dtype=torch.int64)
        )

    def update_mem(self, feats):
        if comm.get_world_size() > 1:
            all_feats = comm.all_gather(feats)
            feats = torch.cat([feat_.to(self.memory.device) for feat_ in all_feats])
        left_mem = self.mem_len - self.memory_next[self.task_count].item()
        if left_mem < feats.shape[0]:
            self.memory[self.task_count][self.memory_next[self.task_count] :] = feats[
                :left_mem
            ]
            self.memory[self.task_count][: feats.shape[0] - left_mem] = feats[left_mem:]
            self.memory_next[self.task_count] = feats.shape[0] - left_mem
        else:
            self.memory[self.task_count][
                self.memory_next[self.task_count] : self.memory_next[self.task_count]
                + feats.shape[0]
            ] = feats
            self.memory_next[self.task_count] = (
                self.memory_next[self.task_count] + feats.shape[0]
            ) % self.mem_len

    def sample_mem(self, task_id):
        if comm.get_world_size() > 1:
            if comm.get_rank() == 0:
                sample = torch.randint(
                    0, self.mem_len, (self.mem_batch,), device=self.memory.device
                )
            else:
                # NOTE scatter raises errors
                sample = None
            sample = comm.all_gather(sample)[0].to(self.memory.device)
        else:
            sample = torch.randint(
                0, self.mem_len, (self.mem_batch,), device=self.memory.device
            )
        return self.memory[task_id][sample]

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        # assume queries are same across layers
        B, nL, C = x_query.shape
        if train:
            self.update_mem(x_query[:, 0, :])
            if self.task_count > 0:
                mem_samples = [self.sample_mem(i) for i in range(self.task_count)]
        p_return = []
        p_loss = 0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here

                # cosine similarity to match keys/querries
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_query[:, l, :], dim=1).detach()
                cos_sim = torch.einsum("bj,kj->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())
                    loss = (
                        1.0
                        - cos_sim[
                            :,
                            self.task_count
                            * self.top_k : (self.task_count + 1)
                            * self.top_k,
                        ]
                    ).sum()  # pos
                    p_loss += loss * self.loss_weight
                    if self.task_count > 0:
                        # neg
                        loss = (cos_sim[:, : self.task_count * self.top_k]).sum()
                        p_loss += loss * self.loss_weight_neg
                        for ti, ti_samp in enumerate(mem_samples):
                            q = nn.functional.normalize(ti_samp, dim=1).detach()
                            cos_sim = torch.einsum("bj,kj->bk", q, n_K)
                            # pos
                            loss = (
                                1.0
                                - cos_sim[:, ti * self.top_k : (ti + 1) * self.top_k]
                            ).sum()
                            p_loss += loss * self.loss_weight_mem_pos
                            # neg
                            if ti == 0:
                                loss = 0.0
                            else:
                                loss = (cos_sim[:, : ti * self.top_k]).sum()
                            loss += (
                                cos_sim[
                                    :,
                                    (ti + 1)
                                    * self.top_k : (self.task_count + 1)
                                    * self.top_k,
                                ]
                            ).sum()
                            p_loss += loss * self.loss_weight_mem_neg
                    P_ = (
                        p[
                            self.task_count
                            * self.top_k : (self.task_count + 1)
                            * self.top_k
                        ]
                        .flatten(0, 1)
                        .unsqueeze(0)
                        .expand(B, -1, -1)
                    )  # B, num_e_prompts*topk, d
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    P_ = p[k_idx]
                lp.append(P_)
            # g prompts
            if l in self.g_layers:
                p = getattr(self, f"g_p_{l}")  # 0 based indexing here
                P_ = p.expand(B, -1, -1)
                lp.append(P_)
            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        if self.training:
            return torch.stack(p_return, dim=0), p_loss
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), p_loss


class L2PppMaskMC(L2PppMaskM):
    def _init_smart(self, *args, **kws):
        super(L2PppMaskM, self)._init_smart(*args, **kws)
        self.mem_len = 1024
        self.mem_batch = 64 // comm.get_world_size()
        self.temp = 0.07
        self.register_buffer(
            "memory", torch.zeros(self.n_tasks, self.mem_len, self.key_d)
        )
        self.register_buffer(
            "memory_next", torch.zeros(self.n_tasks, dtype=torch.int64)
        )

    def sample_mem(self, task_id):
        if comm.get_world_size() > 1:
            if comm.get_rank() == 0:
                sample = torch.randint(
                    0,
                    self.mem_len,
                    (self.mem_batch * comm.get_world_size(),),
                    device=self.memory.device,
                )
            else:
                # NOTE scatter raises errors
                sample = None
            sample = comm.all_gather(sample)[0].to(self.memory.device)
        else:
            sample = torch.randint(
                0, self.mem_len, (self.mem_batch,), device=self.memory.device
            )
        return self.memory[task_id][sample]

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        # assume queries are same across layers
        B, nL, C = x_query.shape
        if train:
            self.update_mem(x_query[:, 0, :])
            if self.task_count > 0:
                mem_samples = [self.sample_mem(i) for i in range(self.task_count + 1)]
        p_return = []
        p_loss = 0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here

                # cosine similarity to match keys/querries
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_query[:, l, :], dim=1).detach()
                cos_sim = torch.einsum("bj,kj->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())

                    if self.task_count == 0:
                        # ref to BYOL
                        loss = (
                            2.0
                            - 2
                            * cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ]
                        ).sum() / B  # pos only
                        p_loss += loss * self.loss_weight
                    else:
                        # ref to SupCon, keys as query for contrastive
                        # 1. cur domain
                        # NOTE no self similarity
                        logits_pos = (
                            cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ].T
                            / self.temp
                        )
                        q_neg_n = nn.functional.normalize(
                            torch.cat(mem_samples[: self.task_count], dim=0), dim=1
                        )
                        logits_full = torch.cat(
                            [
                                logits_pos,
                                torch.einsum(
                                    "kj,bj->kb",
                                    n_K[
                                        self.task_count
                                        * self.top_k : (self.task_count + 1)
                                        * self.top_k
                                    ],
                                    q_neg_n,
                                )
                                / self.temp,
                            ],
                            dim=1,
                        )
                        loss = (
                            torch.log(torch.exp(logits_full).sum(dim=1))
                            - logits_pos.mean(dim=1)
                        ).sum()
                        p_loss += loss * self.loss_weight
                        # 2. previous domain
                        for t in range(self.task_count):
                            tn_K = nn.functional.normalize(
                                K[t * self.top_k : (t + 1) * self.top_k, :], dim=1
                            )
                            q_pos_n = nn.functional.normalize(
                                mem_samples[t][
                                    self.mem_batch
                                    * comm.get_local_rank() : self.mem_batch
                                    * (comm.get_local_rank() + 1)
                                ],
                                dim=1,
                            )
                            q_neg_n = nn.functional.normalize(
                                torch.cat(
                                    mem_samples[:t] + mem_samples[t + 1 :], dim=0
                                ),
                                dim=1,
                            )
                            logits_pos = (
                                torch.einsum("kj,bj->kb", tn_K, q_pos_n) / self.temp
                            )
                            logits_full = torch.cat(
                                [
                                    logits_pos,
                                    torch.einsum("kj,bj->kb", tn_K, q_neg_n)
                                    / self.temp,
                                ],
                                dim=1,
                            )
                            loss = (
                                torch.log(torch.exp(logits_full).sum(dim=1))
                                - logits_pos.mean(dim=1)
                            ).sum()
                            p_loss += loss * self.loss_weight
                    P_ = (
                        p[
                            self.task_count
                            * self.top_k : (self.task_count + 1)
                            * self.top_k
                        ]
                        .flatten(0, 1)
                        .unsqueeze(0)
                        .expand(B, -1, -1)
                    )  # B, num_e_prompts*topk, d
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    P_ = p[k_idx]
                lp.append(P_)
            # g prompts
            if l in self.g_layers:
                p = getattr(self, f"g_p_{l}")  # 0 based indexing here
                P_ = p.expand(B, -1, -1)
                lp.append(P_)
            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        if self.training:
            return torch.stack(p_return, dim=0), p_loss
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), p_loss

class L2PppMaskAttn(nn.Module):
    def __init__(
        self,
        emb_d,
        n_tasks,
        pool_size,
        num_prompts,
        num_layers,
        loss_weight,
        topk,
        vis_period,
        key_dim=768,
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(pool_size, num_prompts, num_layers, loss_weight)
        self.vis_period = vis_period
        self.top_k = topk

        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d, ortho=True)
            k = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
            a = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)
            setattr(self, f"e_a_{e}", a)

    def _init_smart(self, pool_size, num_prompts, num_layers, loss_weight):
        # prompt basic param
        self.e_pool_size = int(pool_size)
        self.e_p_length = int(num_prompts)
        self.e_layers = list(range(num_layers))

        # strenth of ortho penalty
        self.ortho_mu = loss_weight[1]

        # strenth of query-key simlarity
        self.cos_mu = loss_weight[0]

    def process_task_count(self, task_id):
        self.task_count = task_id

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        """
        select all layers in one pass
        NOTE assume x_query to be batch_size * num_layer * c
        """
        B, nL, C = x_query.shape
        assert nL == len(self.e_layers)
        # e prompts
        p_return = []
        p_loss = 0
        vis_attn = []
        for l in range(nL):
            K = getattr(self, f"e_k_{l}")
            A = getattr(self, f"e_a_{l}")
            p = getattr(self, f"e_p_{l}")
            pt = self.top_k
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_query[:, l, :], dim=1).detach()
            cos_sim = torch.einsum("bj,kj->bk", q, n_K)
            if train:
                vis_attn.append(cos_sim.detach())
                loss = (
                            1.0
                            - cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ]
                        ).sum()*self.cos_mu
                K = K[s:f]
                A = A[s:f]
                p = p[s:f]
                loss += ortho_penalty(K)*self.ortho_mu
                loss += ortho_penalty(A)*self.ortho_mu
                loss += ortho_penalty(p.flatten(start_dim=1, end_dim=2))*self.ortho_mu
                p_loss+=loss
                K=K.unsqueeze(0).expand(B, -1, -1)
                A=A.unsqueeze(0).expand(B, -1, -1)
                p=p.unsqueeze(0).expand(B, -1, -1,-1)
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                K = K[k_idx]
                A = A[k_idx]
                p = p[k_idx]

            # with attention and cosine sim
            n_K=nn.functional.normalize(K, dim=-1)
            n_A = nn.functional.normalize(A, dim=-1)
            a_k = (n_K*n_A).sum(-1) # b x topk

            P_ = torch.einsum("bk,bkld->bld", a_k, p)
            p_return.append(P_)

        p_return = torch.stack(p_return, dim=0)
        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        return p_return, p_loss





class FixedPrompts(nn.Module):
    def __init__(
        self,
        emb_d,
        num_prompts,
        num_layers,
    ):
        super().__init__()
        self.emb_d = emb_d
        self.e_layers = list(range(num_layers))
        self.e_p_length = int(num_prompts)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_p_length, emb_d)
            setattr(self, f"e_p_{e}", p)

    def process_task_count(self, task_id):
        self.task_count = task_id

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        """
        select all layers in one pass
        NOTE assume x_query to be batch_size * num_layer * c
        """
        B, nL, C = x_query.shape
        assert nL == len(self.e_layers)
        # e prompts
        p_return = []
        p_loss = torch.tensor(0, dtype=x_query.dtype, device=x_query.device)
        for l in range(nL):
            p = getattr(self, f"e_p_{l}")  # 0 based indexing here
            p_return.append(p.unsqueeze(0).expand(B, -1, -1))
        p_return = torch.stack(p_return, dim=0)
        return p_return, p_loss


class FixedPromptsTaskInc(nn.Module):
    def __init__(
        self,
        n_tasks,
        emb_d,
        num_prompts,
        num_layers,
    ):
        super().__init__()
        self.emb_d = emb_d
        self.e_layers = list(range(num_layers))
        self.e_p_length = int(num_prompts)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(n_tasks, self.e_p_length, emb_d)
            setattr(self, f"e_p_{e}", p)

    def process_task_count(self, task_id):
        self.task_count = task_id

    def forward(self, nL, task_id):
        """
        select all layers in one pass
        NOTE assume x_query to be batch_size * num_layer * c
        """
        assert nL == len(self.e_layers)
        # e prompts
        p_return = []
        for l in range(nL):
            p = getattr(self, f"e_p_{l}")[task_id]  # 0 based indexing here
            p_return.append(p)
        p_return = torch.stack(p_return, dim=0)
        return p_return


class SPrompts(nn.Module):
    def __init__(
        self,
        emb_d,
        n_tasks,
        n_keys,
        keys_path,
        num_prompts,
        num_layers,
        key_dim=768,
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.n_prompts=num_prompts
        self.n_keys=n_keys
        self.n_layers=num_layers
        # prompt / key init
        for e in range(self.n_layers):
            p = tensor_prompt(self.n_tasks,num_prompts, emb_d)
            k = torch.zeros(self.n_tasks, n_keys, self.key_d)
            setattr(self, f"e_p_{e}", p)
            self.register_buffer(f"e_k_{e}", k)
        self.task_keys_temp=torch.load(keys_path,map_location=self.e_k_0.device)
        self.load_key=False

    def process_task_count(self, task_id):
        self.task_count = task_id

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        if not self.load_key:
            for e in range(self.n_layers):
                k = getattr(self,f"e_k_{e}")
                k[self.task_count]=self.task_keys_temp.to(k.device)
            self.load_key=True
        B, nL, C = x_query.shape
        p_loss = 0.0
        lp = []
        for l in range(nL):
            # prompts
            p = getattr(self, f"e_p_{l}")  # 0 based indexing here
            k=getattr(self, f"e_k_{l}")[:self.task_count+1]
            if train:
                P_ = p[self.task_count].unsqueeze(0).expand(B, -1, -1)# B, num_prompts, d
            else:
                q=x_query[:,l]
                dists= ((q.unsqueeze(1).unsqueeze(1) - k.unsqueeze(0))**2).sum(-1) #B x C - T x L x C -> B x T x L
                dists_fl=dists.flatten(1,2) # B x TL
                min_dists_idx=torch.argmin(dists_fl,dim=1)
                match_tasks=min_dists_idx // self.n_keys
                P_=p[match_tasks]
            lp.append(P_)
        
        return torch.stack(lp, dim=0), p_loss


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(*dims, ortho=False):
    p = torch.nn.Parameter(torch.FloatTensor(*dims), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


def tensor_prompt_value(*dims, ortho=False):
    return tensor_prompt(*dims, ortho=ortho).data

def build_stage_prompt_pool(prompt_cfg,stage_num_prompts,stage_idx,emb_d,num_layers,key_dim,vis_period):
        if stage_num_prompts == 0:
            prompt_stage=nn.Identity() # trivial impl
        else:
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = L2Ppp(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = L2PppMask(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskOrth":
                prompt_stage = L2PppMaskOrth(
                    orth_mu=prompt_cfg.ORTH_MU,
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskSeOrth":
                prompt_stage = L2PppMaskSeOrth(
                    orth_mu=prompt_cfg.ORTH_MU,
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskSeOrthWD":
                prompt_stage = L2PppMaskSeOrthWD(
                    orth_mu=prompt_cfg.ORTH_MU,
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskSeOrthTD":
                prompt_stage = L2PppMaskSeOrthTD(
                    td_mu=prompt_cfg.TD_MU,
                    orth_mu=prompt_cfg.ORTH_MU,
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskSeOrthTO":
                prompt_stage = L2PppMaskSeOrthTO(
                    to_mu=prompt_cfg.TO_MU,
                    orth_mu=prompt_cfg.ORTH_MU,
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = L2PppMaskAttn(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = L2PppMaskBs(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = L2PppMask2(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = L2PppMaskM(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = L2PppMaskMC(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            elif prompt_cfg.PROMPT_TYPE == "DualPromptL2P":
                stage_type=prompt_cfg.POOL_TYPE[stage_idx]
                if stage_type=="e":
                    prompt_stage = L2PppMask(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
                else:
                    prompt_stage = FixedPrompts(
                        emb_d=emb_d,
                        num_prompts=stage_num_prompts*prompt_cfg.TOP_K,
                        num_layers=num_layers,
                        )
                    
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = FixedPrompts(
                    emb_d=emb_d,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                )
            elif prompt_cfg.PROMPT_TYPE == "SPrompts":
                prompt_stage = SPrompts(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    n_keys=prompt_cfg.NUM_KEYS,
                    keys_path=prompt_cfg.KEYS_PATH,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    key_dim=key_dim,
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = CodaPromptWd(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            else:  # CODAPrompt
                prompt_stage = CodaPrompt(
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRENT_TASK)
        return prompt_stage