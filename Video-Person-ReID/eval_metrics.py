from __future__ import print_function, absolute_import
import numpy as np
import copy

import os.path as osp
from os import mkdir

def dump_matches_imgids(output_dir, matches_imgids):
    if not osp.isdir(output_dir):
        mkdir(output_dir)
    for q_imgid, g_imgids in matches_imgids.iteritems():
        with open(osp.join(output_dir, '%s.txt' % q_imgid), 'w') as f:
            for g_imgid in g_imgids:
                f.write('%s\n' % g_imgid)

def dump_query_result(output_dir, matches_imgids, top_N=100):
    if not osp.isdir(output_dir):
        mkdir(output_dir)
    with open(osp.join(output_dir, 'track2.txt'), 'w') as f:
        for q_imgid, g_imgids in sorted(matches_imgids.iteritems()):
            g_imgids = [str(imgid) for imgid in g_imgids]
            if top_N > 0:
                g_imgids = g_imgids[:top_N]
            st = ' '.join(g_imgids)
            f.write(st + '\n')

def evaluate_imgids(distmat, q_pids, g_pids, q_camids, g_camids, q_imgids, g_imgids, max_rank=50, top_N=0):
    '''
    mAP and cmc in per-image basis
    g_imgids, g_imgids: list of list of imgid
    return all_cmc, mAP, and matches_imgids (map from q_imgids to g_imgids)
    '''

    num_q, num_g = distmat.shape

    assert(len(q_imgids) == num_q and len(g_imgids) == num_g)

    q_counts = [len(imgs) for imgs in q_imgids]
    g_counts = [len(imgs) for imgs in g_imgids]
    num_gi = sum(g_counts)
    #print('num_q = %d, num_g = %d, num_gi = %d' % (num_q, num_g, num_gi))

    if num_gi < max_rank:
        max_rank = num_gi
        print("Note: number of gallery samples is quite small, got {}".format(num_gi))
    indices = np.argsort(distmat, axis=1)
    # count gt and prediction (first imgid only)
    matches_gt_pred = {}
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_imgid = q_imgids[q_idx][0]
        matches_gt_pred[(q_pid, q_imgid)] = ([], [])
        for gi_idx in range(num_g):
            g_idx = indices[q_idx, gi_idx]
            g_pid = g_pids[g_idx]
            g_imgid = g_imgids[g_idx][0]
            matches_gt_pred[(q_pid, q_imgid)][1].append((g_pid, g_imgid))
            if g_pid == q_pid:
                matches_gt_pred[(q_pid, q_imgid)][0].append((g_pid, g_imgid))
    # expand to per-gallery image
    indices_expanded = np.zeros((num_q, num_gi), dtype=np.int32)
    for q_idx in range(num_q):
        pos = 0
        for s_idx in range(num_g):
            g_idx = indices[q_idx][s_idx]
            g_count = g_counts[g_idx]
            indices_expanded[q_idx][pos:pos+g_count] = g_idx
            pos += g_count
    indices = indices_expanded
    # create matches_imgids from indices_expanded
    matches_imgids = {}
    for q_idx in range(num_q):
        matches_imgids[q_imgids[q_idx][0]] = []
        g_poss = [0] * num_g
        for gi_idx in range(num_gi):
            g_idx = indices_expanded[q_idx][gi_idx]
            #print('q_idx = %d, gi_idx = %d, g_idx = %d' % (q_idx, gi_idx, g_idx))
            #print('g_poss = ' + str(g_poss))
            matches_imgids[q_imgids[q_idx][0]].append(g_imgids[g_idx][g_poss[g_idx]])
            g_poss[g_idx] += 1
        if top_N > 0:
            matches_imgids[q_imgids[q_idx][0]] = matches_imgids[q_imgids[q_idx][0]][:top_N]
        #print(str(q_imgids[q_idx][0]) + ': ' + str(matches_imgids[q_imgids[q_idx][0]]))

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # find false positive result
    matches_imgids_FP = {}
    top_FP = 3
    for q_idx in range(num_q):
        matches_imgids_FP[q_imgids[q_idx][0]] = []
        FPs = []
        for gi_idx in range(min(top_FP, num_g)):
            if matches[q_idx, gi_idx] == 0:
                FPs.append(indices[q_idx, gi_idx])

        g_poss = [0] * num_g
        for gi_idx in range(num_gi):
            g_idx = indices_expanded[q_idx][gi_idx]
            if g_idx in FPs:
                #print('q_idx = %d, gi_idx = %d, g_idx = %d' % (q_idx, gi_idx, g_idx))
                #print('g_poss = ' + str(g_poss))
                matches_imgids_FP[q_imgids[q_idx][0]].append(g_imgids[g_idx][g_poss[g_idx]])
                g_poss[g_idx] += 1

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        #keep += True ###### keep everything

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if top_N == 0:
            AP = tmp_cmc.sum() / num_rel
        else:
            AP = tmp_cmc[:top_N].sum() / num_rel
        all_AP.append(AP)

        #print('%s %s AP: %f,  cmc[0]: %f' % (q_pids[q_idx], q_imgids[q_idx], AP, cmc[0]))
        #if AP < cmc[0]:
        #    print(orig_cmc[:top_N])

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, matches_imgids, matches_imgids_FP, matches_gt_pred



def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, top_N=0):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        #keep += True ###### keep everything

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if top_N == 0:
            AP = tmp_cmc.sum() / num_rel
        else:
            AP = tmp_cmc[:top_N].sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


