#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
probFea: all feature vectors of the query set, shape = (image_size, feature_dim)
galFea: all feature vectors of the gallery set, shape = (image_size, feature_dim)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy # for KL divergence
from math import log

def compute_metadata_distance_hard(q_metadatas, g_metadatas, metadata_prob_ranges):
    q_num = q_metadatas.shape[0]
    g_num = g_metadatas.shape[0]
    dist = np.zeros((q_num, g_num), dtype=np.float32)
    for iq in range(q_num):
        for ig in range(g_num):
            for p_begin, p_end in metadata_prob_ranges:
                cq = np.argmax(q_metadatas[iq][p_begin:p_end])
                cg = np.argmax(g_metadatas[ig][p_begin:p_end])
                if cq != cg:
                    dist[iq, ig] += 1
                    break
    return dist

def compute_metadata_distance_semihard(q_metadatas, g_metadatas, metadata_prob_ranges):
    q_num = q_metadatas.shape[0]
    g_num = g_metadatas.shape[0]
    dist = np.zeros((q_num, g_num), dtype=np.float32)
    for iq in range(q_num):
        for ig in range(g_num):
            for p_begin, p_end in metadata_prob_ranges:
                cq = np.argmax(q_metadatas[iq][p_begin:p_end])
                cg = np.argmax(g_metadatas[ig][p_begin:p_end])
                if cq != cg and cq != (p_end - p_begin - 1) and cg != (p_end - p_begin - 1): # the last class is "other"
                    dist[iq, ig] += 1
                    break
    return dist

def compute_metadata_distance_easy(q_metadatas, g_metadatas, metadata_prob_ranges):
    q_num = q_metadatas.shape[0]
    g_num = g_metadatas.shape[0]
    dist = np.ones((q_num, g_num), dtype=np.float32)
    for iq in range(q_num):
        for ig in range(g_num):
            for p_begin, p_end in metadata_prob_ranges:
                cq = np.argmax(q_metadatas[iq][p_begin:p_end])
                cg = np.argmax(g_metadatas[ig][p_begin:p_end])
                if cq == cg:
                    dist[iq, ig] = 0
                    break
    return dist

def compute_KL_divergence(q_metadatas, g_metadatas, metadata_prob_ranges = [(0,6), (6,18), (18,26)]):
    q_num = q_metadatas.shape[0]
    g_num = g_metadatas.shape[0]
    m_num = len(metadata_prob_ranges)
    KL_div = np.zeros((q_num, g_num, m_num), dtype=np.float32)
    epsilon = 1e-4
    for iq in range(q_num):
        for ig in range(g_num):
            for im, (p_begin, p_end) in enumerate(metadata_prob_ranges):
                KL_div[iq, ig, im] = entropy(q_metadatas[iq][p_begin:p_end]+epsilon, g_metadatas[ig][p_begin:p_end]+epsilon)
    return KL_div

def compute_pred(metadatas, metadata_prob_ranges):
    all_num = metadatas.shape[0]
    m_num = len(metadata_prob_ranges)
    pred = np.zeros((all_num, m_num), dtype=np.int32)
    for im, (p_begin, p_end) in enumerate(metadata_prob_ranges):
        pred[:,im] = np.argmax(metadatas[:,p_begin:p_end], axis=1)
    return pred


def compute_confusion_weight_old(q_pred, g_pred, confusion_mat):
    q_num = q_pred.shape[0]
    g_num = g_pred.shape[0]
    c_num = confusion_mat.shape[0]

    confusion_mat = confusion_mat + 1e-4*np.ones((c_num, c_num), dtype=np.float32)

    c_weight = np.transpose(confusion_mat)*np.diag(confusion_mat)
    c_weight += np.transpose(c_weight)
    c_sum = np.sum(confusion_mat, axis=0).reshape(1,-1)
    c_sum = np.matmul(np.transpose(c_sum), c_sum)
    c_weight = c_weight * np.reciprocal(c_sum)
    #c_weight[range(c_num),range(c_num)]/=2
    np.fill_diagonal(c_weight, 1) # no penalty for the same class
    #print('c_weight = ')
    #print(c_weight)
    
    confusion_weight = np.ones((q_num, g_num), dtype=np.float32)
    for iq in range(q_num):
        for ig in range(g_num):
            confusion_weight[iq, ig] = c_weight[q_pred[iq], g_pred[ig]]
    return confusion_weight

    
def compute_confusion_weight(q_pred, g_pred, confusion_mat):
    q_num = q_pred.shape[0]
    g_num = g_pred.shape[0]
    c_num = confusion_mat.shape[0]

    #print('confusion_mat = ')
    #print(confusion_mat)
    confusion_mat = confusion_mat + 1e-4*np.ones((c_num, c_num), dtype=np.float32)
    c_sum = np.sum(confusion_mat, axis=0)
    confusion_mat_norm = confusion_mat * np.reciprocal(c_sum)
    #print('confusion_mat_norm = ')
    #print(confusion_mat_norm)
    c_weight = np.matmul(np.transpose(confusion_mat_norm), confusion_mat_norm)
    np.fill_diagonal(c_weight, 1) # no penalty for the same class
    #print('c_weight = ')
    #print(c_weight)
    
    confusion_weight = np.ones((q_num, g_num), dtype=np.float32)
    for iq in range(q_num):
        for ig in range(g_num):
            confusion_weight[iq, ig] = c_weight[q_pred[iq], g_pred[ig]]
    return confusion_weight
    

def cluster_gallery_soft(gf, g_metadatas, metadata_prob_ranges = [(0,6), (6,18), (18,26)], k=20, learning_rate=0.5, num_iter=20, MemorySave=False, Minibatch=2000):
    '''
    return new gallery feature gf_new
    '''
    gf = gf.copy() # make a copy since it will be updated in each iteration
    g_num = gf.shape[0]
    # meta data penalty
    '''dist_meta = np.zeros((g_num, g_num), dtype=np.float16)
    epsilon = 1e-4
    for i in range(g_num):
        metaI = g_metadatas[i]
        for j in range(g_num):
            metaJ = g_metadatas[j]
            for prob_range_begin, prob_ranges_end in metadata_prob_ranges:
                if entropy (metaI[prob_range_begin:prob_ranges_end] + epsilon, metaJ[prob_range_begin:prob_ranges_end] + epsilon) > 0.5:
                    dist_meta[i][j] = 1
                    break'''
    dist_meta = compute_metadata_distance_hard(g_metadatas, g_metadatas, metadata_prob_ranges)
    for iter in range(num_iter):
        #print('iter: %d' % iter)
        #print('computing original distance')
        if MemorySave:
            g_g_dist = np.zeros(shape=[g_num, g_num], dtype=np.float16)
            i = 0
            while True:
                it = i + Minibatch
                if it < np.shape(gf)[0]:
                    g_g_dist[i:it, ] = np.power(cdist(gf[i:it, ], gf), 2).astype(np.float16)
                else:
                    g_g_dist[i:, :] = np.power(cdist(gf[i:, ], gf), 2).astype(np.float16)
                    break
                i = it
        else:
            g_g_dist = cdist(gf, gf).astype(np.float16)
            g_g_dist = np.power(g_g_dist, 2).astype(np.float16)
        dist_min = np.min(g_g_dist[np.triu_indices(g_num,1)])
        dist_max = np.max(g_g_dist[np.triu_indices(g_num,1)])
        #print('dist_min = %f, dist_max = %f' % (dist_min, dist_max))
        #g_g_dist = np.transpose(g_g_dist / np.max(g_g_dist, axis=0))
        # apply meta data
        g_g_dist += np.transpose(dist_meta * np.max(g_g_dist, axis=1))
        initial_rank = np.argsort(g_g_dist).astype(np.int32)
        # apply mean field
        gf_new = gf.copy()
        sigma = dist_min / 2 + 1
        for i in range(g_num):
            k_neigh_index = initial_rank[i, :k + 1]
            sigma = np.min(g_g_dist[i, k_neigh_index[1:]]) + 1
            weight = np.exp(-g_g_dist[i, k_neigh_index] / sigma)
            weight /= np.sum(weight)
            if i % 100 == 0 and False:
                print(i)
                print(k_neigh_index)
                print(g_g_dist[i, k_neigh_index])
                print(weight)
            gf_new[i] = np.dot(np.transpose(gf[k_neigh_index]), weight)
        gf = gf * (1 - learning_rate) +  gf_new * (learning_rate)
    return gf


def re_ranking_metadata_soft_v3(original_dist, metadata_dist, query_num, all_num, r_metadata, k1, k2, lambda_value):
    '''
    input:
        original_dist: pre-compute distmat
        metadata_dist: metadata distance
        r_metadata: weight for metadata distance
    return:
        final_dist
    '''
    
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    gallery_num = all_num
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    ### additional scaling
    scaling = False
    if scaling:
        tmp_rank = np.argsort(original_dist).astype(np.int32)
        min_dist = original_dist[range(all_num), tmp_rank[:,1]]
        metadata_dist = np.transpose(metadata_dist * min_dist)
        #print('min_dist = ')
        #print(min_dist)
    ###
    original_dist += r_metadata * metadata_dist

    
    print('starting re_ranking')
    initial_rank = np.argsort(original_dist).astype(np.int32)
    V = np.zeros_like(original_dist).astype(np.float16)
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]

    # np.save('final_dist.npy', final_dist)

    return final_dist




def re_ranking_metadata_soft_v2(qf, gf, q_metadatas, g_metadatas, confusion_mats, metadata_prob_ranges, k1=4, k2=4, lambda_value=0.5, MemorySave=False, Minibatch=2000):

    m_num = len(metadata_prob_ranges)
    for p_begin, p_end in metadata_prob_ranges:
        assert (p_begin, p_end) in confusion_mats

    query_num = qf.shape[0]
    all_num = query_num + gf.shape[0]
    feat = np.append(qf, gf, axis=0)
    all_metadatas = np.append(q_metadatas, g_metadatas, axis=0)
    ###feat = np.concatenate((feat, all_metadatas*20), axis=1)
    # feat = np.append(probFea, galFea)
    # feat = np.vstack((probFea, galFea))
    feat = feat.astype(np.float16)
    print('computing original distance')
    if MemorySave:
        original_dist = np.zeros(shape=[all_num, all_num], dtype=np.float16)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it, ] = np.power(cdist(feat[i:it, ], feat), 2).astype(np.float16)
            else:
                original_dist[i:, :] = np.power(cdist(feat[i:, ], feat), 2).astype(np.float16)
                break
            i = it
    else:
        original_dist = cdist(feat, feat).astype(np.float16)
        original_dist = np.power(original_dist, 2).astype(np.float16)
    del feat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)

    # apply meta data
    print('computing KL divergence')
    
    KL_div = compute_KL_divergence(all_metadatas, all_metadatas, metadata_prob_ranges)
    KL_div_U = compute_KL_divergence(all_metadatas, np.ones(all_metadatas.shape, dtype=np.float32), metadata_prob_ranges)
    conf_pred = np.zeros((all_num, all_num, m_num), dtype=np.float32)
    for im, (p_begin, p_end) in enumerate(metadata_prob_ranges):
        conf_pred[:,:,im] = KL_div_U[:,:,im] * np.transpose(KL_div_U[:,:,im]) / (np.log(p_end - p_begin)*np.log(p_end - p_begin))
    pred = compute_pred(all_metadatas, metadata_prob_ranges)
    confusion_dist = np.zeros((all_num, all_num, m_num), dtype=np.float32)
    for im, (p_begin, p_end) in enumerate(metadata_prob_ranges):
        confusion_weight = compute_confusion_weight(pred[:,im], pred[:,im], confusion_mats[(p_begin, p_end)])
        confusion_dist[:,:,im] = -np.log(confusion_weight + 1e-4) / np.log(p_end-p_begin)

    pred_weight = conf_pred * confusion_dist# * KL_div
    pred_weight = np.sum(pred_weight, axis=2)
    #print('confusion_dist = ')
    #print(confusion_dist)

    tmp_rank = np.argsort(original_dist).astype(np.int32)
    min_dist = original_dist[range(all_num), tmp_rank[:,1]]
    #print('min_dist = ')
    #print(min_dist)
    pred_dist = np.transpose(pred_weight * min_dist)
    #print('pred_dist = ')
    #print(pred_dist)

    r_KL = 10#0.5#20.0
    #print('original_dist = ')
    #print(original_dist)
    #original_dist_no_meta = original_dist.copy()
    original_dist += pred_dist*r_KL
    #original_dist = np.clip(original_dist, 0, 1) # not meaningful
    #print('original_dist = ')
    #print(original_dist)

    initial_rank = np.argsort(original_dist).astype(np.int32)
    #original_dist_no_query = original_dist.copy()
    #original_dist_no_query[:,:query_num] = 1000.0
    #initial_rank = np.argsort(original_dist_no_query).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)



    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value

    #original_dist_easy_meta = original_dist_no_meta + 100*compute_metadata_distance_easy(all_metadatas, all_metadatas, metadata_prob_ranges)
    #original_dist_easy_meta = original_dist_easy_meta[:query_num, ]
    #final_dist = jaccard_dist * (1 - lambda_value) + original_dist_easy_meta * lambda_value

    #original_dist_no_meta = original_dist_no_meta[:query_num, ]
    #final_dist = jaccard_dist * (1 - lambda_value) + original_dist_no_meta * lambda_value


    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]

    # np.save('final_dist.npy', final_dist)

    return final_dist


def re_ranking_metadata_soft(qf, gf, q_metadatas, g_metadatas, metadata_prob_ranges, k1=4, k2=4, lambda_value=0.5, MemorySave=False, Minibatch=2000):
    query_num = qf.shape[0]
    all_num = query_num + gf.shape[0]
    feat = np.append(qf, gf, axis=0)
    #meta = np.append(q_metadatas, g_metadatas, axis=0)
    ###feat = np.concatenate((feat, meta*20), axis=1)
    # feat = np.append(probFea, galFea)
    # feat = np.vstack((probFea, galFea))
    feat = feat.astype(np.float16)
    print('computing original distance')
    if MemorySave:
        original_dist = np.zeros(shape=[all_num, all_num], dtype=np.float16)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it, ] = np.power(cdist(feat[i:it, ], feat), 2).astype(np.float16)
            else:
                original_dist[i:, :] = np.power(cdist(feat[i:, ], feat), 2).astype(np.float16)
                break
            i = it
    else:
        original_dist = cdist(feat, feat).astype(np.float16)
        original_dist = np.power(original_dist, 2).astype(np.float16)
    del feat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    # apply meta data
    print('computing KL divergence')
    KL_div = np.zeros((all_num, all_num), dtype=np.float16)
    tmp_rank = np.argsort(original_dist).astype(np.int32)
    for i in range(all_num):
        if i < query_num:
            metaI = q_metadatas[i]
        else:
            metaI = g_metadatas[i - query_num]
        d_min = original_dist[i][tmp_rank[i,1]]
        #print('d_min: %f' % d_min)
        for j in range(all_num):
            if j < query_num:
                metaJ = q_metadatas[j]
            else:
                metaJ = g_metadatas[j - query_num]
            for prob_range_begin, prob_range_end in metadata_prob_ranges:
                hard_threshold = True
                epsilon = 1e-4
                pk = metaI[prob_range_begin:prob_range_end] + epsilon
                qk = metaJ[prob_range_begin:prob_range_end] + epsilon
                if hard_threshold:
                    if np.argmax(pk) != np.argmax(qk):
                        KL_div[i][j] += 100
                        break
                    else:
                        continue
                #s = entropy(pk, qk)*0.5 + entropy(qk, pk)*0.5
                s = min(entropy(pk, qk), entropy(qk, pk))
                #print('%d: %f' % (num_classes, s))
                #KL_div[i][j] += max(s/log(num_classes) - 1, 0) * d_min
                KL_div[i][j] += s * d_min
    print('KL_div min: %f' % np.min(KL_div[np.triu_indices(all_num,1)]))
    print('KL_div max: %f' % np.max(KL_div[np.triu_indices(all_num,1)]))
    r_KL = 1.0
    original_dist = np.clip(original_dist+KL_div*r_KL, 0, 1)
                    


    initial_rank = np.argsort(original_dist).astype(np.int32)
    #original_dist_no_query = original_dist.copy()
    #original_dist_no_query[:,:query_num] = 1000.0
    #initial_rank = np.argsort(original_dist_no_query).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]

    # np.save('final_dist.npy', final_dist)

    return final_dist
