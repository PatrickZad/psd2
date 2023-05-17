import torch
import torch.nn as nn
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import mat_heatmap


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

    def process_task_count(self):
        self.task_count += 1

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
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        return p_return, p_loss


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean() * 1e-6


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

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

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self, f"e_k_{l}")  # 0 based indexing here
            p = getattr(self, f"e_p_{l}")  # 0 based indexing here

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum("bj,kj->bk", q, n_K)

            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry), -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:, k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]

            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, :, i:, :].reshape((B, -1, self.emb_d))

        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length / 2)
            p = getattr(self, f"g_p_{l}")  # 0 based indexing here
            P_ = p.expand(len(x_querry), -1, -1)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(nn.Module):
    def __init__(
        self,
        emb_d,
        n_tasks,
        pool_size,
        num_prompts,
        num_layers,
        topk,
        loss_weight,
        vis_period,
        key_dim=768,
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(pool_size, num_prompts, num_layers, topk)
        self.loss_weight = loss_weight
        self.vis_period = vis_period

        # e prompt init
        self.e_p = tensor_prompt(num_layers, self.e_pool_size, self.e_p_length, emb_d)
        self.e_k = tensor_prompt(num_layers, self.e_pool_size, self.key_d)

    def _init_smart(self, pool_size, num_prompts, num_layers, topk):
        self.top_k = topk

        # prompt locations
        self.e_layers = list(range(num_layers))

        # prompt pool size
        self.e_p_length = int(num_prompts)
        self.e_pool_size = int(pool_size)

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        """
        select all layers in one pass
        NOTE assume x_query to be batch_size * num_layer * c
        """
        B, nL, C = x_query.shape
        assert nL == len(self.e_layers)
        # cosine similarity to match keys/querries
        n_K = nn.functional.normalize(self.e_k.detach(), dim=-1)
        q = nn.functional.normalize(x_query, dim=-1).detach()
        cos_sim = torch.einsum("blj,lkj->blk", q, n_K)
        top_k = torch.topk(cos_sim, self.top_k, dim=-1)
        k_idx = top_k.indices  # batch_size, num_layers, topk
        idx_offset = k_idx.new_tensor([i * self.e_pool_size for i in range(nL)]).view(
            1, -1, 1
        )  # offset afetr flatten
        flattened_k_idx = (k_idx + idx_offset).flatten(1)  # batch_size, num_layers*topk
        P_ = self.e_p.flatten(0, 1)[
            flattened_k_idx
        ]  # batch_size, num_layers*topk, e_p_length, emb_d
        P_ = P_.reshape(B, nL, self.top_k, self.e_p_length, self.emb_d)
        if train:
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    vis_img = mat_heatmap(cos_sim.flatten(1), vmin=-1.0, vmax=1.0)
                    vis_img = (
                        torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1)
                        / 255.0
                    )
                    storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
            selected_K = self.e_k.flatten(0, 1)[flattened_k_idx].reshape(
                B, nL, self.top_k, C
            )  # batch_size, num_layers, topk, qdim
            batched_selected_sim = torch.einsum(
                "blj,blkj->blk", q, nn.functional.normalize(selected_K, dim=-1)
            )
            loss = (1.0 - batched_selected_sim).sum() / B
        else:
            loss = 0
        p_return = P_.transpose(0, 1).reshape((nL, B, -1, self.emb_d))
        return p_return, loss


# NOTE use prompt mask
class L2POrg(nn.Module):
    def __init__(
        self,
        emb_d,
        n_tasks,
        pool_size,
        num_prompts,
        num_layers,
        topk,
        loss_weight,
        vis_period,
        key_dim=768,
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(pool_size, num_prompts, num_layers, topk)
        self.loss_weight = loss_weight
        self.vis_period = vis_period

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)

    def _init_smart(self, pool_size, num_prompts, num_layers, topk):
        self.top_k = topk

        # prompt locations
        self.e_layers = list(range(num_layers))

        # prompt pool size
        self.e_p_length = int(num_prompts)
        self.e_pool_size = int(pool_size)

    def process_task_count(self):
        self.task_count += 1

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
            K = getattr(self, f"e_k_{l}")  # 0 based indexing here
            p = getattr(self, f"e_p_{l}")  # 0 based indexing here

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_query[:, l], dim=1).detach()
            cos_sim = torch.einsum("bj,kj->bk", q, n_K)
            if train:
                vis_attn.append(cos_sim.detach())
                start = self.task_count * self.top_k
                end = (self.task_count + 1) * self.top_k
                single_prompt_mask = torch.arange(start, end).to(x_query.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(B, -1)
                P_ = p[prompt_mask].reshape(B, self.top_k * self.e_p_length, -1)
                selected_k = n_K[prompt_mask]  # B, top_k, key_d
                sim = selected_k * q.unsqueeze(1)
                reduced_loss = -torch.sum(sim) / q.shape[0]
                p_loss += reduced_loss
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx].reshape(B, self.top_k * self.e_p_length, -1)
            p_return.append(P_)
        p_return = torch.stack(p_return, dim=0)
        if self.vis_period > 0:
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


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(*dims, ortho=False):
    p = torch.nn.Parameter(torch.FloatTensor(*dims), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p
