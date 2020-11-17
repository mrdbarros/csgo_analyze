import logging

import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        return torch.sum(torch.round(torch.sigmoid(output)) == target).float() / output.shape[0]






def generic_forwardpass(batch,model,loss_fn,device,**kwargs):
    x_input = batch[:-1]
    y = batch[-1]

    y = y.to(device)

    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(*[x_tensor.to(device) for x_tensor in x_input], **kwargs)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)

    return loss,y_pred,y




def generic_backwardpass(loss,optimizer,scheduler):
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    scheduler.step()

def train_cycle(loss_fn, model, train_data_loader, optimizer,scheduler,device,**kwargs):
    model.train()
    model.tab_model.eval()
    model.image_model.eval()
    for t, batch in enumerate(train_data_loader):
        loss,_,_ = generic_forwardpass(batch, model, loss_fn,device,**kwargs)
        if (t + 1) % 5 == 0:
            logging.info("%s / %s = %s", t, len(train_data_loader), loss.item())
        generic_backwardpass(loss, optimizer, scheduler)

def validation_cycle(valid_data_loader, model, loss_fn, tensorboard_class,device,**kwargs):
    model.eval()
    valid_loss = AverageMeter('Loss', ':.4e')
    valid_accuracy = AverageMeter('Acc', ':6.2f')
    for t, batch in enumerate(valid_data_loader):

        with torch.no_grad():
            loss,y_pred,y = generic_forwardpass(batch,model,loss_fn,device,**kwargs)
            if t % 5 == 0:
                logging.info("%s / %s = %s", t, len(valid_data_loader), loss.item())
            acc = accuracy(y_pred, y)
            valid_accuracy.update(acc.item(), y.shape[0])
            valid_loss.update(loss.item(), y.shape[0])


    logging.info(' * Acc {valid_accuracy.avg:.3f} Loss {valid_loss.avg:.3f}'
          .format(valid_accuracy=valid_accuracy, valid_loss=valid_loss))
    tensorboard_class.writer.add_scalar("loss:",
                                        valid_loss.avg, tensorboard_class.i)
    tensorboard_class.writer.add_scalar("accuracy:",
                                        valid_accuracy.avg, tensorboard_class.i)
    tensorboard_class.i += 1


def one_epoch(loss_fn, model, train_data_loader, valid_data_loader, optimizer, tensorboard_class,scheduler,device,**kwargs):

    train_cycle(loss_fn,model,train_data_loader,optimizer,scheduler,device,**kwargs)
    validation_cycle(valid_data_loader, model, loss_fn, tensorboard_class,device,**kwargs)


def one_epoch_singleimage(loss_fn, model, train_data_loader, valid_data_loader, optimizer, tensorboard_class,device):
    model.train()
    valid_loss = AverageMeter('Loss', ':.4e')
    valid_accuracy = AverageMeter('Acc', ':6.2f')
    for t, batch in enumerate(train_data_loader):

        x_input = batch[:-1]
        y = batch[-1]

        y = y.to(device)
        for i, x_tensor in enumerate(x_input):
            x_input[i] = x_tensor.to(device)
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(*x_input)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 30 == 0:
            logging.info("%s / %s = %s",t,  len(train_data_loader), loss.item())


        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        # scheduler.step()

    model.eval()
    for t, batch in enumerate(valid_data_loader):

        x_input = batch[:-1]
        y = batch[-1]
        # Forward pass: compute predicted y by passing x to the model.
        y = y.to(device)
        for i, x_tensor in enumerate(x_input):
            x_input[i] = x_tensor.to(device)
        with torch.no_grad():
            y_pred = model(*x_input)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            if t % 30 == 0:
                logging.info("%s / %s = %s", t, len(valid_data_loader), loss.item())
            acc = accuracy(y_pred, y)
            valid_accuracy.update(acc.item(), y.shape[0])
            valid_loss.update(loss.item(), y.shape[0])

    logging.info(' * Acc {valid_accuracy.avg:.3f} Loss {valid_loss.avg:.3f}'
          .format(valid_accuracy=valid_accuracy, valid_loss=valid_loss))

    tensorboard_class.writer.add_scalar("loss:",
                                        valid_loss.avg, tensorboard_class.i)
    tensorboard_class.writer.add_scalar("accuracy:",
                                        valid_accuracy.avg, tensorboard_class.i)
    tensorboard_class.i += 1

class TensorboardClass():
    def __init__(self, writer):
        self.i = 0
        self.writer = writer