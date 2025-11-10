from torchvision import transforms
from utils import *

from datetime import datetime

from configs.lion_pytorch import Lion

class setting_config:
    """
    the config of training setting.
    """

    network = 'msdaunet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [8,16,24,32,48,64],
        'bridge': True,
        'gt_ds': True,
    }

    datasets = 'isic18'
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets in not right!')

    criterion = GT_BceDiceLoss(wb=1, wd=1)

    pretrained_path = './pre_trained/'
    order=1
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 8
    epochs = 300
    accuracy_list = []
    work_dir = 'results/' + datasets + '_batch_size:'+ str(batch_size) +',epochs:'+ str(epochs) +',优化器：Adamw' ', lr:0.002,权重1:2,编码器23:swinT,解码器2:mewb'+'/'

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD', 'Lion'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty)
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.002 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.9 # default: 0 – momentum factor
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum
    elif opt == 'Lion':
        lr = 0.002  # default: 1e-4 – learning rate
        betas = (0.9, 0.99)  # default: (0.9, 0.99) – coefficients used for computing running averages of gradient and its square
        weight_decay = 0  # default: 0 – weight decay coefficient


# 设置不同学习率调度策略（Learning Rate Scheduler）的参数

    sch = 'CosineAnnealingLR'
    if sch == 'StepLR': # 学习率衰减周期（step_size）、衰减系数（gamma）和上一个周期的索引（last_epoch）
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR': # 学习率衰减的里程碑（milestones）、衰减系数（gamma）和上一个周期的索引（last_epoch）
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR': # 学习率衰减系数（gamma）和上一个周期的索引（last_epoch）
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.


    elif sch == 'CosineAnnealingLR': # 最大迭代次数（T_max）、最小学习率（eta_min）和上一个周期的索引（last_epoch）
        T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.


    elif sch == 'ReduceLROnPlateau':  # 模式（mode，min或max）、学习率降低因子（factor）、耐心值（patience，无改进的周期数）、阈值（threshold，用于测量新最优值的变化）、阈值模式（threshold_mode，相对或绝对）、冷却时间（cooldown，恢复正常操作前的周期数）、最小学习率（min_lr）和小衰减量（eps）
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.


    elif sch == 'CosineAnnealingWarmRestarts': # 第一个重启的迭代次数（T_0）、重启后增加T_{i}的因子（T_mult）、最小学习率（eta_min）和上一个周期的索引（last_epoch）
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR': # 预热周期数（warm_up_epochs）、衰减系数（gamma）和学习率衰减的里程碑（milestones）
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR': # 预热周期数（warm_up_epochs）
        warm_up_epochs = 20
