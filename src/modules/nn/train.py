import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import optuna

from .model import Model


class NNDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device):
        self.x, self.y = x, y
        self.device = device
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, i):
        data = (self.x[0][i].to(self.device), self.x[1][i].to(self.device))
        return data, self.y[i].to(self.device)

def train(model, train_loader, test_loader, device):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = F.binary_cross_entropy
    epochs = 30

    def preprocess(y):
        return torch.round(y[0]), y[1]

    precision = ignite.metrics.Precision(preprocess, average=False)
    recall = ignite.metrics.Recall(preprocess, average=False)
    F1 = (precision * recall * 2 / (precision + recall)).mean()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': ignite.metrics.Accuracy(preprocess),
                 'f1': F1,
                 'cross_entropy': ignite.metrics.Loss(criterion)},
        device=device)
    writer = SummaryWriter()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        i = (engine.state.iteration - 1) % len(train_loader) + 1
        if i % 500 == 0:
            print(f"Epoch[{engine.state.epoch}] Iteration[{i}/{len(train_loader)}] "
                  f"Loss: {engine.state.output:.2f}")
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    def write_metrics(metrics, writer, mode: str, epoch: int):
        """print metrics & write metrics to log"""
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['cross_entropy']
        avg_f1 = metrics['f1']
        print(f"{mode} Results - Epoch: {epoch}  "
              f"Avg accuracy: {avg_accuracy:.3f} Avg loss: {avg_nll:.3f} "
              f"Avg F1: {avg_f1:.3f}")
        writer.add_scalar(f"{mode}/avg_loss", avg_nll, epoch)
        writer.add_scalar(f"{mode}/avg_accuracy", avg_accuracy, epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        write_metrics(metrics, writer, 'training', engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        write_metrics(metrics, writer, 'validation', engine.state.epoch)

    handler = ModelCheckpoint(dirname='./checkpoints', filename_prefix='sample',
                              n_saved=5, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})

    handler = EarlyStopping(
        patience=5,
        score_function=lambda x: x.state.metrics['f1'],
        trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)
    return handler.best_score, model
