import torch.optim.lr_scheduler as lr_scheduler


class WarmUpScheduler(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, 
                 warmup_steps,
                 last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step)/float(warmup_steps)
            else :
                return 1

        super(WarmUpScheduler, self).__init__(optimizer,lr_lambda,last_epoch=last_epoch)

class LinearPerEpochScheduler(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, 
                 warmup_steps,
                 last_epoch=-1):
        def lr_lambda(step):
            step = step % warmup_steps
            return float(step)/float(warmup_steps)

        super(LinearPerEpochScheduler, self).__init__(optimizer,lr_lambda,last_epoch=last_epoch)


