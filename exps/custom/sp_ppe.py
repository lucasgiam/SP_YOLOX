# encoding: utf-8
import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp


# Root directory of dataset
datadir = "./datasets/sp_ppe"


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------- model config ------------ #
        self.num_classes = 8                    # num of classes
        self.seed = 1993                        # set seed for training
        self.depth = 0.33                       # yolo-s = 0.33, yolo-m = 0.67, yolo-l = 1.00
        self.width = 0.50                       # yolo-s = 0.50, yolo-m = 0.75, yolo-l = 1.00
        self.warmup_epochs = 1                  # num of epochs used for warmup
        self.max_epoch = 100                    # max num of epochs used for training     
        self.basic_lr_per_img = 0.001 / 16.0    # learning rate for one image (during training, lr will multiply batchsize)
        self.scheduler = "yoloxwarmcos"         # name of learning rate scheduler
        self.weight_decay = 0.01                # weight decay of optimizer
        self.momentum = 0.9                     # momentum of optimizer
        self.data_num_workers = 2               # num of workers
        self.input_size = (640, 640)            # input image size (height, width)
        self.test_size = (640, 640)             # output image size during evaluation/testing
        self.nmsthre = 0.50                     # nms threshold during evaluation/testing
        self.print_interval = 1                 # log period in iter, for example, if set to 1, user could see log every iteration
        self.eval_interval = 1                  # eval period in epoch, for example, if set to 1, model will be evaluate after every epoch
        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0                  # prob of applying mosaic aug
        self.mixup_prob = 1.0                   # prob of applying mixup aug
        self.hsv_prob = 1.0                     # prob of applying hsv aug
        self.flip_prob = 0.5                    # prob of applying flip aug
        self.degrees = 10.0                     # rotation angle range in degrees, for example, if set to 2.0, the true range is (-2.0, 2.0)
        self.translate = 0.1                    # translation range in fraction, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.scale = (0.1, 2.0)                 # scaling range
        self.mosaic_scale = (0.8, 1.6)          # mosaic scaling range
        self.shear = 2.0                        # shear angle range, for example, if set to 2.0, the true range is (-2.0, 2.0)
        self.enable_mixup = True                # to apply mixup aug or not
        # ---------------------------------------- #
        self.perspective = 0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        print("num_classes: ", self.num_classes)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = VOCDetection(
                data_dir=os.path.join(datadir, "VOCdevkit"),
                image_sets=[('2012', 'train')],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import VOCDetection, ValTransform

        valdataset = VOCDetection(
            data_dir=os.path.join(datadir, "VOCdevkit"),
            image_sets=[('2012', 'val')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator