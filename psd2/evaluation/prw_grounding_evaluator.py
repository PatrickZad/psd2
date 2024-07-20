import logging
from .grounding_evaluator import GroundingEvaluator
import time
import torch
from torchvision.ops.boxes import box_area

logger = logging.getLogger(__name__)  # setup_logger()


class PrwGroundingEvaluator(GroundingEvaluator):
    #TODO cross cam eval
    def process(self, inputs, model):
        """
        Args:
            inputs:
                a batch of
                {
                    "image": augmented image tensor
                    "instances": an Instances object with attrs
                        {
                            image_size: hw (int, int)
                            file_name: full path string
                            image_id: filename string
                            gt_boxes: Boxes
                            gt_classes: tensor full with 0s
                            gt_pids: person identity tensor in [-1, max_id]
                            org_img_size: hw before augmentation (int, int)
                            org_gt_boxes: Boxes before augmentation
                        }
                }
            model:
                grounding model to run the specialized inference method
        """
        outputs=model(inputs,self.q_feats.repeat(len(inputs),1,1))
        """
        outputs:
                a batch of instances with attrs
                {
                    pred_boxes: Boxes in augmentation range (resizing/cropping/flipping/...)
                    pred_scores: tensor
                }
        """
        equal_compute_time_stamp=time.perf_counter()

        n_query=self.qid.shape[0]
        cur_grounding_trues=[]
        cur_grounding_scores=[]
        cur_num_gt=torch.zeros(n_query)
        cur_num_pos=torch.zeros(n_query)
        # process
        for bi in range(len(inputs)):
            out_boxes=outputs[bi].pred_boxes.tensor.to(self._cpu_device)
            out_scores=outputs[bi].pred_scores.to(self._cpu_device)
            gt_boxes=inputs[bi]["instances"].gt_boxes.tensor
            gt_pids=inputs[bi]["instances"].gt_pids
            qg_match_mask=self.qid.unsqueeze(1)==gt_pids[None]

            has_gt_mask=qg_match_mask.sum(-1)
            cur_num_gt+=has_gt_mask
            cur_grounding_scores.append(out_scores)

            if has_gt_mask.sum()==0: # pure negative
                qg_match_trues=torch.zeros(n_query,dtype=torch.long)
            else:
                qg_match_idx=torch.argmax(qg_match_mask.long(),dim=1)
                qg_match_gt_boxes=gt_boxes[qg_match_idx]
                qg_match_ious=pairwise_iou(qg_match_gt_boxes,out_boxes)
                qg_match_trues=torch.zeros_like(qg_match_ious,dtype=torch.long)
                qg_match_trues[qg_match_ious>=0.5]=1
                qg_match_trues[has_gt_mask==0]=0

            cur_grounding_trues.append(qg_match_trues)
            cur_num_pos+=qg_match_trues
        
        cur_grounding_trues=torch.split(torch.stack(cur_grounding_trues,dim=1),1,dim=0)
        cur_grounding_scores=torch.split(torch.stack(cur_grounding_scores,dim=1),1,dim=0)
        cur_num_gt=cur_num_gt.numpy().tolist()
        cur_num_pos=cur_num_pos.numpy().tolist()

        for qi,(qi_trues,qi_scores,qi_gts,qi_pos) in enumerate(zip(cur_grounding_trues,cur_grounding_scores,cur_num_gt,cur_num_pos)):
            self.grounding_trues[qi].extend(qi_trues[0].numpy().tolist())
            self.grounding_scores[qi].extend(qi_scores[0].numpy().tolist())
            self.num_gt[qi]+=qi_gts
            self.num_pos[qi]+=qi_pos
        return equal_compute_time_stamp

def pairwise_iou(boxes1,boxes2):
    assert boxes1.shape[0]==boxes2.shape[0]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2]) 
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:]) 

    wh = (rb - lt).clamp(min=0) 
    inter = wh[:, 0] * wh[:, 1]  # [N,M]

    union = area1 + area2 - inter

    iou = inter / union
    return iou