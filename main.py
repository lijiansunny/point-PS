import torch
import os
import numpy as np
from options import train_opts
from utils import logger, recorders
from models import custom_model, solver_utils, model_utils
from torch import distributed as dist
from collections import OrderedDict

import train_utils
import test_utils

args = train_opts.TrainOpts().parse()
log = logger.Logger(args)
# rank = int(os.environ['RANK'])
# local_rank = int(os.environ['LOCAL_RANK'])
# args.world_size = int(os.environ['WORLD_SIZE'])
# torch.distributed.init_process_group(backend="nccl",world_size=args.world_size,rank=rank)

dist.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
if args.local_rank == 0:
    print(torch.cuda.device_count())  # 打印gpu数量
    print('world_size', dist.get_world_size())  # 打印当前进程数
    # print(os.getcwd())


def main(args):
    if args.local_rank == 0:
        print("=> fetching img pairs in %s" % (args.data_dir))

    if args.dataset == 'PS_Synth_Dataset':
        from datasets.PS_Synth_Dataset import PS_Synth_Dataset
        train_set = PS_Synth_Dataset(args, args.data_dir, 'train')
        val_set = PS_Synth_Dataset(args, args.data_dir, 'val')
    else:
        raise Exception('Unknown dataset: %s' % (args.dataset))

    if args.concat_data:
        if args.local_rank == 0:
            print('****** Using cocnat data ******')
            print("=> fetching img pairs in %s" % (args.data_dir2))

        train_set2 = PS_Synth_Dataset(args, args.data_dir2, 'train')
        val_set2 = PS_Synth_Dataset(args, args.data_dir2, 'val')
        train_set = torch.utils.data.ConcatDataset([train_set, train_set2])
        val_set = torch.utils.data.ConcatDataset([val_set, val_set2])
    if args.local_rank == 0:
        print('\t Found Data: %d Train and %d Val' % (len(train_set), len(val_set)))
        print('\t Train Batch %d, Val Batch: %d' % (args.batch, args.val_batch))
    train_iloss = []
    train_eloss = []
    test_iloss = []
    test_eloss = []

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
                                               num_workers=args.workers, pin_memory=args.cuda, shuffle=False,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch,
                                             num_workers=args.workers, pin_memory=args.cuda, shuffle=False)

    model = custom_model.buildModel(args)
    optimizer, scheduler, records = solver_utils.configOptimizer(args, model)
    criterion = solver_utils.Criterion(args)
    recorder = recorders.Records(args.log_dir, records)

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # 这句莫忘，否则相当于没有shuffle数据
        recorder.insertRecord('train', 'lr', epoch, scheduler.get_last_lr()[0])
        loss_iter0, loss_epoch0 = train_utils.train(args, train_loader, model, criterion, optimizer, log, epoch,
                                                    recorder)
        train_iloss.append(loss_iter0)
        train_eloss.append(loss_epoch0)
        scheduler.step()
        if epoch % args.save_intv == 0 and args.local_rank == 0:
            model_utils.saveCheckpoint(args.cp_dir, epoch, model, optimizer, recorder.records, args)

        if epoch % args.val_intv == 0 and args.local_rank == 0:
            loss_iter1, loss_epoch1 = test_utils.test(args, 'val', val_loader, model, log, epoch, recorder)
            test_iloss.append(loss_iter1)
            test_eloss.append(loss_epoch1)

        # Use a barrier() to make sure that all processes have finished reading the
        # checkpoin
        dist.barrier()
    if args.local_rank == 0:
        np.save('./data/Training/calib/train/train_iter_loss', train_iloss)
        np.save('./data/Training/calib/train/train_epoch_loss', train_eloss)
        np.save('./data/Training/calib/train/test_iter_loss', test_iloss)
        np.save('./data/Training/calib/train/test_epoch_loss', test_eloss)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    local_rank = int(os.environ["LOCAL_RANK"])
    main(args)
