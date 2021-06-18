import cv2
import numpy as np
import copy

def V_std(pred, gt, reverse=True):
    """
    It is the implementation of criterions used for our project, namely v_rand and v_info
    Details can be refered from https://www.frontiersin.org/articles/10.3389/fnana.2015.00142/full

    Input: pred, gt => both numpy arrays, both of which predicts edges as "1"
    Return: v_info, v_rand => 2 scalar
    """
    if reverse:
        # reverse means that we assume the "1"s in the figure as edge instead of cell contents
        pred_label = (pred < 0.5).astype(np.uint8)
        gt_label = (gt < 0.5).astype(np.uint8)
    else:
        pred_label = (pred > 0.5).astype(np.uint8)
        gt_label = (gt > 0.5).astype(np.uint8)

    pred_num, pred_out = cv2.connectedComponents(pred_label, connectivity=4)
    gt_num, gt_out = cv2.connectedComponents(gt_label, connectivity=4)

    p = np.zeros((pred_num+1, gt_num+1))
    for i in range(pred_num+1): 
        # we should allow for the situation where gt=0 and pred<>0 (not take into account in calc)
        # and treat the situation where gt<>0 and pred=0 as false
        tmp_mask = (pred_out==i)
        for j in range(1, gt_num+1):
            p[i][j] = np.logical_and(tmp_mask, gt_out==j).sum()
    
    p[0, 0] = 0
    
    tot_sum = p.sum()
    p = p / tot_sum

    s = p.sum(axis=0)
    t = p.sum(axis=1)

    sum_p_log = (p * np.log(p+1e-9)).sum()
    sum_s_log = (s * np.log(s+1e-9)).sum()
    sum_t_log = (t * np.log(t+1e-9)).sum()

    v_info = -2 * (sum_p_log - sum_s_log - sum_t_log) / (sum_s_log  + sum_t_log)

    sum_p_s = (p*p).sum()
    sum_s_s = (s*s).sum()
    sum_t_s = (t*t).sum()
    v_rand = 2 * sum_p_s / (sum_t_s + sum_s_s)
    return v_info, v_rand




if __name__ == '__main__':
    try_img = np.ones((4, 4), dtype=np.uint8)
    # 1 1 1 1
    # 1 1 0 0
    # 0 0 0 1
    # 1 1 1 0
    zeros_places = [(1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (3, 3)]
    for zero_place in zeros_places:
        try_img[zero_place] = 0

    label_img = copy.deepcopy(try_img)
    label_img[1, 2] = 1
    # 1 1 1 1
    # 1 1 0 0
    # 0 0 0 1
    # 1 1 1 0

    # output_img = np.ones((4, 4))
    # num_labels, labels_im = cv2.connectedComponents(try_img, connectivity=4)
    # print(labels_im)
    res = V_std(try_img, label_img, reverse=False)
    print(res)
    