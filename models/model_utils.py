import os
import torch
import torch.nn as nn

def getInput(args, data):
    input_list = [data['input']]
    if args.in_light: input_list.append(data['l'])
    return input_list

def parseData(args, sample, timer=None, split='train'):
    input, target, mask = sample['img'], sample['N'], sample['mask'] 
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        input  = input.cuda(args.local_rank); target = target.cuda(args.local_rank); mask = mask.cuda(args.local_rank)

    input_var  = torch.autograd.Variable(input) #input.requires_grad = False，默认false
    target_var = torch.autograd.Variable(target)
    mask_var   = torch.autograd.Variable(mask, requires_grad=False)

    if timer: timer.updateTime('ToGPU')
    data = {'input': input_var, 'tar': target_var, 'm': mask_var}

    if args.in_light:
        light = sample['light'].expand_as(input)
        if args.cuda: light = light.cuda(args.local_rank)
        light_var = torch.autograd.Variable(light)
        data['l'] = light_var
    return data 

def getInputChanel(args):
    if args.local_rank == 0:
        print('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        if args.local_rank == 0:
            print('[Network Input] Adding Light direction as input')
        c_in += 3
    if args.local_rank == 0:
        print('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        # print(f"work dir is {os.getcwd()}")
        # print(f"current dir is {os.path.dirname(__file__)}")
        checkpoint = torch.load(path)
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        #  checkpoint = torch.load(checkpoint_path, map_location=torch.cuda())
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    # from collections import OrderedDict
    # state_dict = checkpoint['state_dict']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = 'module.'+k  # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records, 
            'args': args}
    torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_%d_rec.pth.tar' % (epoch)))

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    # print('Conv pad = %d' % (pad))
    if batchNorm:
        # print('=> convolutional layer with batchnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                # nn.LeakyReLU(0.1, inplace=True)
                nn.GELU()
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                # nn.LeakyReLU(0.1, inplace=True)
                nn.GELU()
                )

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.LeakyReLU(0.1, inplace=True)
            nn.GELU()
            )
def pixel_shuffle(cin):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(cin, cin * 4, kernel_size=3, stride=1, padding=0),
        # nn.LeakyReLU(0.1, inplace=True),
        nn.GELU(),
        nn.PixelShuffle(upscale_factor=2)
    )
def upsampling(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cin * 4, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.PixelShuffle(upscale_factor=2),
        # nn.Upsample(scale_factor=2, mode='bilinear'),
        # nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1)
        nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)

    )


if __name__=='__main__':
    # checkpoint = torch.load('PS-FCN_B_S_32.pth.tar')
    # print(checkpoint)
    import numpy as np

    x = torch.randn((1, 3, 16, 16))
    y = torch.randn(1, 64, 128, 128)
    # G = FSRNet(hmaps_ch=0, pmaps_ch=10)
    # R = RFDB(64, 32)


