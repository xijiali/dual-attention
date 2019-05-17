from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import os

import torch
from torchvision.transforms import ToTensor
import torchvision
import numpy as np
from PIL import Image
import cv2

from .evaluation_metrics import map_cmc
from .utils.meters import AverageMeter

from .utils import to_torch
from visdom import Visdom
viz = Visdom()
assert viz.check_connection()


def extract_cnn_feature(model, inputs, output_feature=None):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = to_torch(inputs)
    inputs = inputs.to(device)
    outputs,attention = model(inputs, output_feature)
    outputs = outputs.data.cpu()
    attention=attention.data.cpu()
    return outputs,attention


def extract_features(model, data_loader, print_freq=1, output_feature=None,flag=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    Attentions=OrderedDict()

    file_lst=[]

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        file_lst.extend(fnames)
        data_time.update(time.time() - end)

        outputs,attentions = extract_cnn_feature(model, imgs, output_feature)
        for fname, output, pid,attention in zip(fnames, outputs, pids,attentions):
            features[fname] = output
            labels[fname] = pid
            Attentions[fname]=attention

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    if flag is True:
        return features, labels , file_lst ,Attentions
    else:
        return features, labels


def pairwise_distance(query_features, gallery_features, query=None, gallery=None,flag=False):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0) # [3368,2048]
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    # We use clamp to keep numerical stability
    dist = torch.clamp(dist, 1e-8, np.inf)
    if flag is True:
        return dist,x,y
    else:
        return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20),
                 flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Evaluation
    mAP, all_cmc = map_cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, all_cmc[k - 1]))
    if flag is True:
        return mAP,query_ids,query_cams,gallery_ids,gallery_cams
    else:
        return mAP

    # Traditional evaluation
    # Compute mean AP
    # mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    # print('Mean AP: {:4.1%}'.format(mAP))
    #
    # # Compute CMC scores
    # cmc_configs = {
    #     'market1501': dict(separate_camera_set=False,
    #                        single_gallery_shot=False,
    #                        first_match_break=True)}
    # cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
    #                         query_cams, gallery_cams, **params)
    #               for name, params in cmc_configs.items()}
    #
    # print('CMC Scores')
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}'
    #           .format(k, cmc_scores['market1501'][k - 1]))
    #
    # return cmc_scores['market1501'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature=None):
        query_features, _ = extract_features(self.model, query_loader, 1, output_feature)
        gallery_features, _ = extract_features(self.model, gallery_loader, 1, output_feature)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)

    def save(self, query_loader, gallery_loader, query, gallery, output_feature=None):
        query_features, _ ,query_lst, query_attentions= extract_features(self.model, query_loader, 1, output_feature,flag=True)
        print("query list is:{}".format(query_lst))
        gallery_features, _ , gallery_lst, gallery_attentions= extract_features(self.model, gallery_loader, 1, output_feature,flag=True)
        # attentions
        qa = torch.cat([query_attentions[f].unsqueeze(0) for f, _, _ in query], 0)
        m,c1,h1,w1=qa.size()
        ga=torch.cat([gallery_attentions[f].unsqueeze(0) for f, _, _ in gallery], 0)
        n,c2,h2,w2=ga.size()

        distmat,qf,gf = pairwise_distance(query_features, gallery_features, query, gallery,flag=True)
        mAP,query_ids,query_cams,gallery_ids,gallery_cams = evaluate_all(distmat, query=query, gallery=gallery,flag=True)
        assert(len(query_lst)==len(query_ids)==len(query_cams)==qf.size(0)==qa.size(0))
        assert (len(gallery_lst)==len(gallery_ids)==len(gallery_cams)==gf.size(0)==ga.size(0))
        # save
        result = {'gallery_f': gf,
                  'gallery_label': gallery_ids,
                  'gallery_cam': gallery_cams,
                  'query_f': qf,
                  'query_label': query_ids,
                  'query_cam': query_cams,
                  'gallery_path_lst': gallery_lst,
                  'query_path_lst': query_lst,
                  'qa':qa,
                  'ga':ga,
                  }
        save_dir = "/home/xiaoxi.xjl/my_code/UDA/Dual-attention-transfer/logs-2019-0515-1730/feature.pt"
        torch.save(result, save_dir)

        return mAP

    def visualize(self, query_loader, gallery_loader, query, gallery, output_feature=None):
        feature_file = "/home/xiaoxi.xjl/my_code/UDA/Dual-attention-transfer/logs-2019-0515-1730/feature.pt"
        result = torch.load(feature_file)
        query_feature = result['query_f']
        query_cam = result['query_cam']
        query_label = result['query_label']
        gallery_feature = result['gallery_f']
        gallery_cam = result['gallery_cam']
        gallery_label = result['gallery_label']
        gallery_path_lst = result['gallery_path_lst']
        query_path_lst = result['query_path_lst']
        qa=result['qa']
        ga=result['ga']
        root='/home/xiaoxi.xjl/re_id_dataset/Market-1501-v15.09.15'
        for i in range(0,100):
            # visualize query image
            query_path = query_path_lst[i]
            # query_path=query_path[0:-4]
            print("query path:{}".format(query_path))
            query_path=os.path.join(root,'query',query_path)
            img = self.convert(query_path)

            qf = query_feature[i, :].unsqueeze(0)
            query_attention=qa[i,:,:,:]
            print("query_attention size:{}".format(query_attention.size()))#[1,8,4]
            query_attention=self.convert_(query_attention)
            masked_img=0.5*img+0.5*query_attention
            masked_img=torchvision.utils.make_grid(masked_img)

            m = 1
            n = gallery_feature.size(0)
            q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                       torch.pow(gallery_feature, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            q_g_dist.addmm_(1, -2, qf, gallery_feature.t())
            distmat = q_g_dist
            print("distmat size:{}".format(distmat.size()))

            print("Computing CMC and mAP")
            print("query_labels are:{}".format(query_label))
            print("query_label[i] is:{}".format(query_label[i]))
            _, mAP, index = self.eval_func_gpu(distmat, query_label[i], gallery_label, query_cam[i], gallery_cam)
            print("mAP:{}".format(mAP))
            if mAP < 0.05:
                viz.image(img,
                          opts=dict(title=str(i)),
                          )
                viz.image(masked_img,opts=dict(title=str(i)))

                # print top 10
                for j in range(10):
                    gallery_path = gallery_path_lst[index[0,j]]
                    # gallery_path=gallery_path[0:-4]
                    print("gallery path:{}".format(gallery_path))
                    gallery_path=os.path.join(root,'bounding_box_test',gallery_path)
                    gallery_img = self.convert(gallery_path)
                    viz.image(gallery_img,
                              opts=dict(title=str(j)),
                              )
                    gallery_attention=ga[index[0,j],:,:,:]
                    gallery_attention=self.convert_(gallery_attention)
                    masked_gallery=0.5*gallery_img+gallery_attention*0.5
                    masked_gallery=torchvision.utils.make_grid(masked_gallery)
                    viz.image(masked_gallery,
                              opts=dict(title=str(j)),
                              )
                    print("top{} gallery img path is:{}".format(j, gallery_path))

        return

    def convert(self,path):
        img = self.read_image(path)
        img = img.resize((128, 256))
        img = ToTensor()(img)
        return img

    def convert_(self,mask):
        mask=torch.mean(mask,dim=0)
        mask = mask.cpu().numpy()
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask = np.uint8(mask * 255)
        mask = cv2.resize(mask, (128, 256))
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        #mask = cv2.applyColorMap(mask, cv2.COLORMAP_BONE)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = ToTensor()(mask)
        return mask

    def read_image(self,img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        return img

    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        print("num_q:{}".format(num_q))
        print("num_g:{}".format(num_g))

        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        print("indices:{}".format(indices))
        print("indices size:{}".format(indices.size()))


        g_pids=torch.Tensor(g_pids)
        print("g_pids size:{}".format(g_pids.size()))
        print("q_pids:{}".format(q_pids))


        g_camids=torch.Tensor(g_camids)

        matches = g_pids[indices] == q_pids
        keep = ~((g_pids[indices] == q_pids) & (g_camids[indices]  == q_camids))
        #keep = g_camids[indices]  != q_camids.view([num_q, -1])

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank+1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item(),indices

