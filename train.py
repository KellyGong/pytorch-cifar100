import os
from conf import settings
from utils import get_network, get_cv_generator, get_test_dataloader, get_train_dataloader
from trainer import Trainer
import nni


def train(args):
    # preprocessing
    cifar100_cv_generator = get_cv_generator(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    #
    cifar100_train_loader = get_train_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    #
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model = get_network(args)

    trainer = Trainer(model, args)

    y_correct = trainer.cross_validation(cifar100_cv_generator)

    trainer.train(cifar100_train_loader)

    acc, _, _ = trainer.valid(cifar100_test_loader)

    nni.report_final_result(acc)

    trainer.save_result(checkpoint_path, y_correct, {'acc': acc})
