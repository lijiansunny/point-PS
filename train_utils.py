from models import model_utils
from utils import time_utils


def train(args, loader, model, criterion, optimizer, log, epoch, recorder):
    model.train()
    if args.local_rank == 0:
        print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync)
    loss_iter = []
    loss_epoch = []

    for i, sample in enumerate(loader):
        # print(f'data size is:{len(sample)}')
        data = model_utils.parseData(args, sample, timer, 'train')
        input = model_utils.getInput(args, data)

        out_var = model(input)
        # input_var = data['tar']
        # print(f'input size is {input_var.shape}, out_var size is {out_var.shape}')
        timer.updateTime('Forward')

        optimizer.zero_grad()
        loss = criterion.forward(out_var, data['tar'])
        timer.updateTime('Crit')
        criterion.backward()
        timer.updateTime('Backward')

        recorder.updateIter('train', loss.keys(), loss.values())

        optimizer.step()
        timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split': 'train', 'epoch': epoch, 'iters': iters, 'batch': len(loader),
                   'timer': timer, 'recorder': recorder}
            loss_iter.append(log.printItersSummary(opt))

    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    loss_epoch.append(log.printEpochSummary(opt))
    return loss_iter, loss_epoch
