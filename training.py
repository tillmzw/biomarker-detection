#!/usr/bin/env python3

import logging
import datetime
import time
import tempfile
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import average_precision_score

import validator
import utils

logger = logging.getLogger(__name__)


class Trainer():

    def __init__(self, epochs=1):
        """Initialize a training class.
        `epochs`: the number of itertations to train for
        """
        super().__init__()
        self._epochs = epochs
        self._es_best_loss = np.inf
        self._es_waiting = 0

    def get_optimizer(self, model):
        raise NotImplementedError

    def get_loss_function(self, weights=None):
        return nn.BCEWithLogitsLoss(weight=weights)

    def get_validation_loss_function(self, weights=None):
        return nn.BCELoss(weight=weights)

    def get_lr_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=6)

    def get_lr(self, optimizer):
        # borrowed from torch.optim.lr_scheduler
        lrs = [float(group['lr']) for group in optimizer.param_groups]
        if len(lrs) > 1:
            logger.debug(f"Averaging learning rates - have {len(lrs)} entries")
        return np.mean(lrs)

    def should_early_stop(self, loss: float, delta=0.0, patience=3) -> bool:
        if loss < (self._es_best_loss - delta):
            # new loss is better than previous loss
            self._es_best_loss = loss
            self._es_waiting = 0
        else:
            # new loss is equal or worse than previous loss
            self._es_waiting += 1
            logger.info(f"New loss ({loss:.3f}) is equal/worse than previous one ({self._es_best_loss:.3f}); "
                        f"waiting for {self._es_waiting}/{patience} epochs until recommending early stop")

        return self._es_waiting > patience

    def train(self, model, dataloader, state_file=None, validation_dataloader=None):
        """
        Apply this classes optimizer and loss function to `model` and `dataloader`.
        The final model can be saved to `state_file`.
        Validation is performed if `validation_dataloader` is provided at every `validation_rel_step` (in percent of total runs).
        """
        # TODO: this only works for one GPU!
        model_device = next(model.parameters()).device

        # we have a very unbalanced data set so we need to add weight to the loss function
        if hasattr(dataloader.dataset, "class_weights"):
            weights = dataloader.dataset.class_weights()
            if weights is not None:
                logger.info("Applying weights: %s" % ", ".join(
                    ("%d: %.2f" % (i, w) for i, w in enumerate(weights))
                ))
            weights = weights.to(model_device)
        else:
            logger.warning("No class weight calculation supported by data loader %s" % dataloader.dataset.__class__)
            weights = None
        loss_func = self.get_loss_function(weights=weights)
        valid_loss_func = self.get_validation_loss_function(weights=weights)
        optimizer = self.get_optimizer(model)
        lr_sched = self.get_lr_scheduler(optimizer)

        self.get_lr(optimizer)

        wandb.config.update({
            "host": utils.hostname(),
            "git": utils.git_hash(),
            "epochs": self._epochs,
            "batch_size": dataloader.batch_size,
            "n_training_batches": len(dataloader),
            "n_validation_batches": len(validation_dataloader) if validation_dataloader else -1,
        })
        if hasattr(model, "descriptor") and isinstance(model.descriptor, dict):
            wandb.config.update(model.descriptor)

        step = 0
        for epoch in range(self._epochs):  # loop over the dataset multiple times
            # set the model to training mode
            model.train()
            logger.info("Training iteration %d/%d" % (epoch + 1, self._epochs))
            training_start = time.time()
            for i, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, filenames, labels]
                names, coords, images, masks = data

                images = images.to(model_device)
                masks = masks.to(model_device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(images)
                # make sure the labels are on the same device as the data
                loss = loss_func(outputs, masks)
                loss.backward()
                optimizer.step()

                avg_precision = average_precision_score(y_true=masks.detach().to("cpu").numpy(),
                                                        y_score=outputs.detach().to("cpu").numpy())

                step += 1
                wandb.log({"training_loss": loss.item(), "training_avg_precision": avg_precision}, step=step)

            # seconds -> float
            tt = time.time() - training_start
            # minutes with seconds as decimal
            ett = tt // 60 + ((tt % 60) / 60)
            # normalize by number of samples, i.e. normalize to seconds per image
            ett_image = ett * 60 / (dataloader.batch_size * len(dataloader))
            logger.info("Training iteration took %.2f minutes (~ %.0f seconds), or %.4f seconds per image" % (ett, ett * 60, ett_image))
            wandb.log({"epoch_training_time_abs": ett}, step=step)
            wandb.log({"epoch_training_time_image": ett_image}, step=step)

            # start validation for the current epoch
            if validation_dataloader:
                validation_start = time.time()
                try:
                    validation_acc, validation_loss, avg_precision, confusion, prc, roc = validator.validate(model,
                                                                                                             validation_dataloader,
                                                                                                             loss_func=self.get_validation_loss_function(weights))
                    # adapt the learning rate
                    lr_sched.step(avg_precision)
                except Exception as e:
                    logger.error("While validating during training, an error occured:")
                    logger.exception(e)
                else:
                    wandb.log({"validation_accuracy": validation_acc,
                               "validation_loss": validation_loss,
                               "validation_avg_precision": avg_precision,
                               "lr": self.get_lr(optimizer)
                               }, step=step)
                    logger.info(f"Validation during training at step {step}: {validation_acc:.2f}% accuracy, {avg_precision:.2f} avg precision")
                    vt = time.time() - validation_start
                    vtt = vt // 60 + ((vt % 60) / 60)
                    # normalize by number of samples, i.e. normalize to seconds per image
                    vtt_image = vtt * 60 / (dataloader.batch_size * len(dataloader))
                    logger.info("Validation took %.2f minutes (~ %.0f seconds), or %.4f seconds per image" % (vtt, vtt * 60, vtt_image))
                    wandb.log({"epoch_training_validation_time_abs": vtt}, step=step)
                    wandb.log({"epoch_training_validation_time_image": vtt_image}, step=step)

                    # create a plot from the confusion matrix
                    logger.info("Creating confusion matrix plot")
                    plot = utils.plot_confusion_matrix(confusion)
                    implot = utils.plot_to_pil(plot)
                    wandb.log({"epoch_training_confusion_matrix": wandb.Image(implot)}, step=step)

                    logger.info("Creating precision-recall plot")
                    plot = utils.plot_precision_recall(*prc)
                    implot = utils.plot_to_pil(plot)
                    wandb.log({"epoch_training_precision_recall": wandb.Image(implot)})

                    logger.info("Creating roc+auc plot")
                    plot = utils.plot_roc_auc(*roc)
                    implot = utils.plot_to_pil(plot)
                    wandb.log({"epoch_training_roc_auc": wandb.Image(implot)})

                    if self.should_early_stop(validation_loss.item()):
                        logger.warning("Engaging early stopping!")
                        break

            if state_file:
                # save intermediate model
                fname, fext = os.path.basename(state_file).split(".")
                intermed_save = os.path.abspath(os.path.join(state_file, "..", "%s_%04d.%s" % (fname, step, fext)))
                logger.info("Saved intermediate model state file to %s" % intermed_save)
                torch.save(model.state_dict(), intermed_save)

        logger.debug('Finished Training')

        if state_file:
            logger.info('Saving model parameters to %s' % state_file)
            torch.save(model.state_dict(), state_file)


class SGDTrainer(Trainer):
    def get_optimizer(self, model):
        return optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9)


class AdamTrainer(Trainer):
    def get_optimizer(self, model):
        return optim.Adam(params=model.parameters(), lr=1e-4)
