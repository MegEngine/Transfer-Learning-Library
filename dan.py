import time
import shutil
from copy import deepcopy
from types import MethodType

import megengine
import megengine.hub
import megengine.functional as F
from megengine import tensor
from megengine.autodiff import GradManager
from megengine.functional.nn import cross_entropy
from megengine.data import RandomSampler, SequentialSampler, DataLoader
from megengine.optimizer import SGD, LRScheduler

import model as dan_model
import utils
from meter import AverageMeter, ProgressMeter


def load_torch_resnet50_pretrained_dict():
    from torchvision.models.utils import load_state_dict_from_url
    from torchvision.models.resnet import model_urls
    pretrained_dict = load_state_dict_from_url(
        model_urls['resnet50'], progress=True)
    return {k: v.detach().numpy()
        for k, v in pretrained_dict.items()}


def load_backbone():
    model = megengine.hub.load(
        'megengine/models', 'resnet50', pretrained=False)
    pretrained_dict = load_torch_resnet50_pretrained_dict()
    model.load_state_dict(pretrained_dict, strict=True)
    model.out_features = model.fc.in_features
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    model.forward = MethodType(forward, model)
    def copy_head(self):
        return deepcopy(self.fc)
    model.copy_head = MethodType(copy_head, model)
    return model


def main(args):
    train_transform = utils.get_train_transform(
        args.train_resizing,
        random_horizontal_flip=not args.no_hflip,
        random_color_jitter=False,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std)
    val_transform = utils.get_val_transform(
        args.val_resizing,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)

    def get_train_loader(dataset):
        sampler = RandomSampler(dataset,
            batch_size=args.batch_size, drop_last=True)
        return DataLoader(dataset, sampler=sampler,
            transform=train_transform, num_workers=args.workers)
    train_source_loader = get_train_loader(train_source_dataset)
    train_target_loader = get_train_loader(train_target_dataset)
    def get_val_loader(dataset):
        sampler = SequentialSampler(dataset,
            batch_size=args.batch_size, drop_last=False)
        return DataLoader(dataset, sampler=sampler,
            transform=val_transform, num_workers=args.workers)
    val_loader = get_val_loader(val_dataset)
    test_loader = get_val_loader(test_dataset)

    train_source_iter = utils.ForeverDataIterator(train_source_loader)
    train_target_iter = utils.ForeverDataIterator(train_target_loader)

    backbone = load_backbone()
    classifier = dan_model.ImageClassifier(
        backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
        pool_layer=None, finetune=not args.scratch)

    if args.phase == 'test':
        classifier.load_state_dict(megengine.load('classifier.best.ckpt'))
        validate(val_loader, classifier, args)
        return

    grad_manager = GradManager().attach(classifier.parameters())
    optimizer = SGD(classifier.get_parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    class LambdaLR(LRScheduler):
        def __init__(self, optimizer, lr_lambda):
            self.lr_lambda = lr_lambda
            super().__init__(optimizer)
        def get_lr(self):
            return [i * self.lr_lambda(self.current_epoch) for i in self.base_lrs]
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    mkmmd_loss = dan_model.MultipleKernelMaximumMeanDiscrepancy(
        kernels=[dan_model.GaussianKernel(alpha=2**k) for k in range(-3, 2)],
        linear=not args.non_linear)

    best_acc = 0
    latest_ckpt, best_ckpt = 'classifier.latest.ckpt', 'classifier.best.ckpt'
    for epoch in range(args.epochs):
        train(train_source_iter, train_target_iter, classifier, mkmmd_loss,
            grad_manager, optimizer, lr_scheduler, epoch, args)
        acc = validate(val_loader, classifier, args)
        megengine.save(classifier.state_dict(), latest_ckpt)
        if acc > best_acc:
            shutil.copy(latest_ckpt, best_ckpt)
        best_acc = max(acc, best_acc)

    print("best_acc1 = {:3.1f}".format(best_acc))
    classifier.load_state_dict(megengine.load(best_ckpt))
    acc = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc))


def train(train_source_iter, train_target_iter, model, mkmmd_loss,
    grad_manager, optimizer, lr_scheduler, epoch, args):
    '''train for one epoch'''
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':5.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    mkmmd_loss.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)
        x_s, labels_s, x_t, labels_t = map(
            tensor, [x_s, labels_s, x_t, labels_t])
        data_time.update(time.time() - end) # measure data loading time

        with grad_manager:
            y_s, f_s = model(x_s)
            y_t, f_t = model(x_t)
            cls_loss = cross_entropy(y_s, labels_s)
            transfer_loss = mkmmd_loss(f_s, f_t)
            loss = cls_loss + transfer_loss * args.trade_off
            grad_manager.backward(loss)
            optimizer.step().clear_grad()
            lr_scheduler.step() # 已验证和原库一致

        cls_acc = utils.accuracy(y_s, labels_s)
        tgt_acc = utils.accuracy(y_t, labels_t)

        losses.update(loss.item(), x_s.shape[0])
        cls_accs.update(cls_acc.item(), x_s.shape[0])
        tgt_accs.update(tgt_acc.item(), x_t.shape[0])
        trans_losses.update(transfer_loss.item(), x_s.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, args) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    model.eval() # switch to evaluate mode

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images, target = map(tensor, [images, target])
        # compute output
        output = model(images)
        loss = cross_entropy(output, target)

        # measure accuracy and record loss
        acc = utils.accuracy(output, target)

        losses.update(loss.item(), images.shape[0])
        top1.update(acc.item(), images.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    class Args:
        arch = 'resnet50'
        batch_size = 32
        bottleneck_dim = 256
        data = 'Office31'
        epochs = 20
        iters_per_epoch = 500
        log = 'logs/dan/Office31_D2A'
        lr = 0.001 # NOTE lr is 0.003 in pytorch version
        lr_decay = 0.75
        lr_gamma = 0.0003
        momentum = 0.9
        no_hflip = False
        no_pool = False
        non_linear = False
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
        per_class_eval = False
        phase = 'train' # or 'test'
        print_freq = 100
        resize_size = 224
        root = 'data/office31'
        scratch = False
        seed = 0
        source = ['D']
        target = ['A']
        trade_off = 1.0
        train_resizing = 'default'
        val_resizing = 'default'
        wd = 0.0005
        workers = 2
    main(Args)
