import collections
import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader

from datasets.bases import ImageDataset
from utils.meter import AverageMeter
from utils.metrics import R1_mAP
import torchvision.transforms as T
from datasets.preprocessing import RandomErasing
from datasets.sampler import RandomIdentitySampler
from datasets.make_dataloader import train_collate_fn
from utils.metrics import extract_features
from torch.nn import Parameter
from utils.reranking import compute_jaccard_dist
import torch.nn.functional as F
def make_cluster_dataloader(cfg, target_set):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        # T.RandomRotation(12, resample=Image.BICUBIC, expand=False, center=None),
        # T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0),
        #                T.RandomAffine(degrees=0, translate=None, scale=[0.8, 1.2], shear=15, \
        #                               resample=Image.BICUBIC, fillcolor=0)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465))
    ])

    num_workers = cfg.DATALOADER_NUM_WORKERS
    # dataset = Market1501(data_dir=cfg.DATA_DIR, verbose=True)
    # num_classes = dataset.num_train_pids
    train_set = ImageDataset(target_set, train_transforms)

    if cfg.SAMPLER == 'triplet':
        print('using triplet sampler')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.BATCH_SIZE,
                                  num_workers=num_workers,
                                  sampler=RandomIdentitySampler(target_set, cfg.BATCH_SIZE, cfg.NUM_IMG_PER_ID)#,
                                  #collate_fn=train_collate_fn  # customized batch sampler
                                  )
    elif cfg.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  sampler=None,
                                  #collate_fn=train_collate_fn,  # customized batch sampler
                                  drop_last=True
                                  )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    return train_loader

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.MAX_EPOCHS

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    # train
    best_mAP = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            score, feat = model(img, target)
            loss = loss_fn(score, feat, target)

            loss.backward()
            optimizer.step()
            if 'center' in cfg.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        # if epoch % checkpoint_period == 0:
        #     torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    feat = model(img)
                    evaluator.update((feat, vid, camid))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            if mAP>best_mAP:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_best.pth'))



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, \
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, pids, camids, qfeats, gfeats = evaluator.compute()

    np.save(os.path.join(cfg.LOG_DIR, cfg.DIST_MAT) , distmat)
    np.save(os.path.join(cfg.LOG_DIR, cfg.PIDS), pids)
    np.save(os.path.join(cfg.LOG_DIR, cfg.CAMIDS), camids)
    np.save(os.path.join(cfg.LOG_DIR, cfg.IMG_PATH), img_path_list[num_query:])
    torch.save(qfeats, os.path.join(cfg.LOG_DIR, cfg.Q_FEATS))
    torch.save(gfeats, os.path.join(cfg.LOG_DIR, cfg.G_FEATS))

    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)
    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)
    return model

def do_train_dbscan(cfg,
             model,
             target_dataset,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.MAX_EPOCHS

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')
    checkpoint = torch.load(cfg.SOURCE_PRETRAIN_PATH, map_location=torch.device('cpu'))
    copy_state_dict(checkpoint, model)

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    # train
    best_mAP = 0
    target_cluster_dataloader = make_cluster_dataloader(cfg, target_dataset)
    for epoch in range(0, epochs):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        dict_f, _= extract_features(model, target_cluster_dataloader)
        feature_list = torch.stack(list(dict_f.values()))
        print(feature_list.shape)
        rerank_dist = compute_jaccard_dist(feature_list, use_gpu=False).numpy()
        print(rerank_dist.shape)
        if (epoch==0):
            # DBSCAN cluster
            tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            rho = 1.6e-3
            top_num = np.round(rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps for cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)
        print(np.sum((labels==-1)), "images not in the clusters")
        # cfg.num_clusters = num_ids
        print('\n Clustered into {} classes \n'.format(num_ids))

        new_dataset = []
        cluster_centers = collections.defaultdict(list)
        for i, ((fname, _, cid), label) in enumerate(zip(target_dataset, labels)):
            if label==-1: continue
            new_dataset.append((fname,label,cid))
            cluster_centers[label].append(feature_list[i])

        cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers)
        model.classifier.weight.data[:num_ids].copy_(F.normalize(cluster_centers, dim=1).float().cuda())
        train_loader_target = make_cluster_dataloader(cfg, new_dataset)#get_train_loader(dataset_target, args.height, args.width,args.batch_size, args.workers, args.num_instances, iters,                     trainset=new_dataset)
        model.train()
        # # # Optimizer
        # # params = []
        # # for key, value in model.named_parameters():
        # #     if not value.requires_grad:
        # #         continue
        # #     params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        # # optimizer = torch.optim.Adam(params)
        #
        # # Trainer
        # trainer = ClusterBaseTrainer(model, num_cluster=args.num_clusters)
        #
        # train_loader_target.new_epoch()
        #
        # trainer.train(epoch, train_loader_target, optimizer,
        #               print_freq=args.print_freq, train_iters=len(train_loader_target))
        #
        # def save_model(model, is_best, best_mAP):
        #     save_checkpoint({
        #         'state_dict': model.state_dict(),
        #         'epoch': epoch + 1,
        #         'best_mAP': best_mAP,
        #     }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        # model.train()
        # for n_iter, (img, vid) in enumerate(target_cluster_dataloader):
        for iter, (img, pid, camid, img_path) in enumerate(train_loader_target):
            optimizer.zero_grad()
            img = img.to(device)
            target = pid.to(device)
            score, feat = model(img, target)
            loss = loss_fn(score, feat, target)
            loss.backward()
            optimizer.step()
            # if 'center' in cfg.LOSS_TYPE:
            #     for param in center_criterion.parameters():
            #         param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
            #     optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(target_cluster_dataloader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, target_cluster_dataloader.batch_size / time_per_batch))

        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        # if epoch % checkpoint_period == 0:
        #     torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    feat = model(img)
                    evaluator.update((feat, vid, camid))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            if mAP>best_mAP:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_best.pth'))
