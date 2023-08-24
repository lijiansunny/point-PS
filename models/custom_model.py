import torch.nn as nn
from . import model_utils

def buildModel(args):
    if args.local_rank == 0:
        print('Creating Model %s' % (args.model))
    in_c = model_utils.getInputChanel(args)
    other = {'img_num': args.in_img_num, 'in_light': args.in_light}
    if args.model == 'PS_FCN': 
        from models.PS_FCN import PS_FCN
        model = PS_FCN(args.fuse_type, args.use_BN, in_c, other)
    elif args.model == 'PS_FCN_run':
        from models.PS_FCN_run import PS_FCN
        model = PS_FCN(args.fuse_type, args.use_BN, in_c, other)
    else:
        raise Exception("=> Unknown Model '{}'".format(args.model))

    if args.cuda:
        model = model.cuda(args.local_rank)
        if args.use_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 设置多个gpu的BN同步
        model = nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=False,
                                                          broadcast_buffers=False)


    if args.retrain:
        if args.local_rank == 0:
            print("=> using pre-trained model %s" % (args.retrain))
        model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)

    if args.resume:
        if args.local_rank == 0:
            print("=> Resume loading checkpoint %s" % (args.resume))
        model_utils.loadCheckpoint(args.resume, model, cuda=args.cuda)

    if args.local_rank == 0:
        print(model)
        print("=> Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model
