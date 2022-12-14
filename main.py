import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.custom_optimizer import build_optimizer_t2d, T2D_LR_CONFIG, EVL_LR_CONFIG
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
from utils.visualize import visualize_grid_to_grid_with_cls
import time
import numpy as np
import random
try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False
from fvcore.nn import flop_count
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
from models import xclip_notetx
from models.model_t2d import CLIPT2D
from models.model2_steam import EVLTransformer2STREAM
from models.modelevl import EVLTransformer
from models.model_t2d_stream import CLIPT2DS
from models.model_2d_stream import CLIP2DS
from models.model_2d_stream12p import CLIP2DSL
from models.model_2d_stream_lowlevel import CLIP2DSLow
from models.model_2d_stream_cls import CLIP2DSCLS
from models.model_2d_stream_cls_12p import CLIP2DSCLS12P
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--framework', default='XCLIP', type=str)
    parser.add_argument('--backbone_path', type=str)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config, args):
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    if args.framework == 'XCLIP':
        model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
                             device="cpu", jit=False,
                             T=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                             droppath=config.MODEL.DROP_PATH_RATE,
                             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                             use_cache=config.MODEL.FIX_TEXT,
                             logger=logger,
                            )
    elif args.framework == 'XCLIPV':
        model, _ = xclip_notetx.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
                             device="cpu", jit=False,
                             T=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                             droppath=config.MODEL.DROP_PATH_RATE,
                             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                             logger=logger,
                            )
    elif args.framework == 'T2D':
        # model = CLIPT2D(pretrain=args.backbone_path)
        # large
        model = CLIPT2D(num_classes=config.DATA.NUM_CLASSES,
                        pretrain=args.backbone_path,
                        width=config.T2D.WIDTH,
                        patch_size=config.T2D.PATCH_SIZE,
                        layers=config.T2D.LAYERS,
                        heads=config.T2D.HEADS,
                        use_text_classifier=config.T2D.USE_TEXT_CLASSIFIER,
                        use_cls_token=config.T2D.USE_CLS_TOKEN,
                        temporal_model=config.T2D.TEMPORAL_MODEL,
                        text_bias=config.T2D.TEXT_BIAS,
                        text_dim=config.T2D.TEXT_DIM,
                        text_heads=config.T2D.TEXT_HEADS,
                        temporal_layer=config.T2D.TEMPORAL_LAYER,
                        vpt=config.T2D.VPT,
                        text_backbone_name=config.T2D.TEXT_BACKBONE_NAME,
                        attn_type=config.T2D.ATTN_TYPE,
                        use_temporal_cls_token=config.T2D.USE_TEMPORAL_CLS_TOKEN,
                        frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                        input_resolution=config.DATA.INPUT_SIZE,
                        enable_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        imagenet_pretrain=config.T2D.IMAGENET_PRETRAIN,
                        pretrain_jit=config.T2D.PRETRAIN_JIT,
                        )
    elif args.framework == 'T2DS':
        model = CLIPT2DS(num_classes=config.DATA.NUM_CLASSES,
                         pretrain=args.backbone_path,
                         width=config.T2D.WIDTH,
                         patch_size=config.T2D.PATCH_SIZE,
                         layers=config.T2D.LAYERS,
                         heads=config.T2D.HEADS,
                         use_text_classifier=config.T2D.USE_TEXT_CLASSIFIER,
                         temporal_model=config.T2D.TEMPORAL_MODEL,
                         text_bias=config.T2D.TEXT_BIAS,
                         text_dim=config.T2D.TEXT_DIM,
                         text_heads=config.T2D.TEXT_HEADS,
                         temporal_layer=config.T2D.TEMPORAL_LAYER,
                         text_backbone_name=config.T2D.TEXT_BACKBONE_NAME,
                         frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                         input_resolution=config.DATA.INPUT_SIZE,
                         enable_checkpoint=config.TRAIN.USE_CHECKPOINT,
                         imagenet_pretrain=config.T2D.IMAGENET_PRETRAIN,
                         batch_mode=config.T2DS.BATCH_MODE,)
    elif args.framework == '2DS':
        model = CLIP2DS(num_classes=config.DATA.NUM_CLASSES,
                        pretrain=args.backbone_path,
                        width=config.T2D.WIDTH,
                        patch_size=config.T2D.PATCH_SIZE,
                        layers=config.T2D.LAYERS,
                        heads=config.T2D.HEADS,
                        use_text_classifier=config.T2D.USE_TEXT_CLASSIFIER,
                        temporal_model=config.T2D.TEMPORAL_MODEL,
                        text_bias=config.T2D.TEXT_BIAS,
                        text_dim=config.T2D.TEXT_DIM,
                        text_heads=config.T2D.TEXT_HEADS,
                        temporal_layer=config.T2D.TEMPORAL_LAYER,
                        text_backbone_name=config.T2D.TEXT_BACKBONE_NAME,
                        frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                        input_resolution=config.DATA.INPUT_SIZE,
                        enable_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        imagenet_pretrain=config.T2D.IMAGENET_PRETRAIN,
                        batch_mode=config.T2DS.BATCH_MODE, )
    elif args.framework == '2DSCLS':
        model = CLIP2DSCLS(num_classes=config.DATA.NUM_CLASSES,
                        pretrain=args.backbone_path,
                        width=config.T2D.WIDTH,
                        patch_size=config.T2D.PATCH_SIZE,
                        layers=config.T2D.LAYERS,
                        heads=config.T2D.HEADS,
                        use_text_classifier=config.T2D.USE_TEXT_CLASSIFIER,
                        temporal_model=config.T2D.TEMPORAL_MODEL,
                        text_bias=config.T2D.TEXT_BIAS,
                        text_dim=config.T2D.TEXT_DIM,
                        text_heads=config.T2D.TEXT_HEADS,
                        temporal_layer=config.T2D.TEMPORAL_LAYER,
                        text_backbone_name=config.T2D.TEXT_BACKBONE_NAME,
                        frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                        input_resolution=config.DATA.INPUT_SIZE,
                        enable_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        imagenet_pretrain=config.T2D.IMAGENET_PRETRAIN,
                        batch_mode=config.T2DS.BATCH_MODE,
                        use_cls_token=config.T2D.USE_CLS_TOKEN,
                        #visualize=args.visualize,
                           )
    elif args.framework == '2DSCLS12P':
        model = CLIP2DSCLS12P(num_classes=config.DATA.NUM_CLASSES,
                        pretrain=args.backbone_path,
                        width=config.T2D.WIDTH,
                        patch_size=config.T2D.PATCH_SIZE,
                        layers=config.T2D.LAYERS,
                        heads=config.T2D.HEADS,
                        use_text_classifier=config.T2D.USE_TEXT_CLASSIFIER,
                        temporal_model=config.T2D.TEMPORAL_MODEL,
                        text_bias=config.T2D.TEXT_BIAS,
                        text_dim=config.T2D.TEXT_DIM,
                        text_heads=config.T2D.TEXT_HEADS,
                        temporal_layer=config.T2D.TEMPORAL_LAYER,
                        text_backbone_name=config.T2D.TEXT_BACKBONE_NAME,
                        frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                        input_resolution=config.DATA.INPUT_SIZE,
                        enable_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        imagenet_pretrain=config.T2D.IMAGENET_PRETRAIN,
                        batch_mode=config.T2DS.BATCH_MODE,
                        use_cls_token=config.T2D.USE_CLS_TOKEN,
                           )
    elif args.framework == '2DSL':
        model = CLIP2DSL(num_classes=config.DATA.NUM_CLASSES,
                        pretrain=args.backbone_path,
                        width=config.T2D.WIDTH,
                        patch_size=config.T2D.PATCH_SIZE,
                        layers=config.T2D.LAYERS,
                        heads=config.T2D.HEADS,
                        use_text_classifier=config.T2D.USE_TEXT_CLASSIFIER,
                        temporal_model=config.T2D.TEMPORAL_MODEL,
                        text_bias=config.T2D.TEXT_BIAS,
                        text_dim=config.T2D.TEXT_DIM,
                        text_heads=config.T2D.TEXT_HEADS,
                        temporal_layer=config.T2D.TEMPORAL_LAYER,
                        text_backbone_name=config.T2D.TEXT_BACKBONE_NAME,
                        frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                        input_resolution=config.DATA.INPUT_SIZE,
                        enable_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        imagenet_pretrain=config.T2D.IMAGENET_PRETRAIN,
                        batch_mode=config.T2DS.BATCH_MODE, )

    elif args.framework == '2DSLow':
        model = CLIP2DSLow(num_classes=config.DATA.NUM_CLASSES,
                        pretrain=args.backbone_path,
                        width=config.T2D.WIDTH,
                        patch_size=config.T2D.PATCH_SIZE,
                        layers=config.T2D.LAYERS,
                        heads=config.T2D.HEADS,
                        use_text_classifier=config.T2D.USE_TEXT_CLASSIFIER,
                        temporal_model=config.T2D.TEMPORAL_MODEL,
                        text_bias=config.T2D.TEXT_BIAS,
                        text_dim=config.T2D.TEXT_DIM,
                        text_heads=config.T2D.TEXT_HEADS,
                        temporal_layer=config.T2D.TEMPORAL_LAYER,
                        text_backbone_name=config.T2D.TEXT_BACKBONE_NAME,
                        frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
                        input_resolution=config.DATA.INPUT_SIZE,
                        enable_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        imagenet_pretrain=config.T2D.IMAGENET_PRETRAIN,
                        batch_mode=config.T2DS.BATCH_MODE, )


    elif args.framework == 'EVL':
        model = EVLTransformer(
            backbone_name='ViT-B/16-lnpre',
            backbone_type='clip',
            backbone_path=args.backbone_path,
            backbone_mode='finetune',
            decoder_num_layers=4,
            decoder_qkv_dim=768,
            decoder_num_heads=12,
            decoder_mlp_factor=4.0,
            num_classes=400,
            enable_temporal_conv=True,
            enable_temporal_pos_embed=True,
            enable_temporal_cross_attention=True,
            cls_dropout=0.5,
            decoder_mlp_dropout=0.5,
            num_frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
            use_text_classifier=config.T2D.USE_TEXT_CLASSIFIER,
            text_backbone_name=config.T2D.TEXT_BACKBONE_NAME
                        )
    elif args.framework == 'STREAM':
        model = EVLTransformer2STREAM(
            backbone_name='ViT-B/16-lnpre',
            backbone_type='clip',
            backbone_path=args.backbone_path,
            backbone_mode='finetune',
            decoder_num_layers=4,
            decoder_qkv_dim=768,
            decoder_num_heads=12,
            decoder_mlp_factor=4.0,
            num_classes=400,
            enable_temporal_conv=True,
            enable_temporal_pos_embed=True,
            enable_temporal_cross_attention=True,
            cls_dropout=0.5,
            decoder_mlp_dropout=0.5,
            num_frames=config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,
        )
    else:
        raise NotImplementedError

    model = model.cuda()

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                       smoothing=config.AUG.LABEL_SMOOTH,
                                       mixup_alpha=config.AUG.MIXUP,
                                       cutmix_alpha=config.AUG.CUTMIX,
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    if args.framework in ['EVL', 'STREAM']:
        optimizer = build_optimizer_t2d(config, model, EVL_LR_CONFIG)
    elif args.framework in ['T2D', 'T2DS', '2DS','2DSL','2DSLow','2DSCLS','2DSCLS12P']:
        optimizer = build_optimizer_t2d(config, model, T2D_LR_CONFIG)
    else:
        optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    if config.TRAIN.AMP == 'apex':
        assert has_apex
    assert config.TRAIN.AMP in ['apex', 'torch']
    loss_scaler = None
    if config.TRAIN.AMP == 'apex' and config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    else:
        loss_scaler = torch.cuda.amp.GradScaler()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)





    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.MODEL.RESET_EPOCH = False
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger, loss_scaler)
        if config.MODEL.RESET_EPOCH:
            logger.info('reset epoch to 0')
            start_epoch = 0

    text_labels = generate_text(train_data)

    if config.TEST.ONLY_TEST:
        config.defrost()
        config.TEST.NUM_CLIP = 4
        config.TEST.NUM_CROP = 3
        config.freeze()
        train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
        acc1 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        return
    assert (not args.visualize)

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, loss_scaler)

        acc1 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best, loss_scaler, config.SAVE_NUM)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc1 = validate(val_loader, text_labels, model, config)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, loss_scaler=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    texts = text_labels.cuda(non_blocking=True)

    for idx, batch_data in enumerate(train_loader):

        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1,config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN,3)+images.size()[-2:])

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        with torch.cuda.amp.autocast(enabled=config.TRAIN.AMP == 'torch'):
            # ##--add comput flops--
            # fake_input_data = torch.zeros([1, 3, 16, 224, 224], device='cuda')
            # count_dict, *_ = flop_count(model.module, (fake_input_data,texts))
            # count = sum(count_dict.values())
            # logger.info(f"number of GFLOPs: {count}")
            # ##--add comput flops--
            output = model(images, texts)

            total_loss = criterion(output, label_id)
            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.AMP == 'apex' and config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif config.TRAIN.AMP == 'torch':
            loss_scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if config.TRAIN.AMP == 'torch':
                    loss_scaler.step(optimizer)
                    loss_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            if config.TRAIN.AMP == 'torch':
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()

    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES * config.DATA.CLIP_LEN
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                if args.visualize:
                    output,attn = model(image_input, text_inputs)
                    visualize_grid_to_grid_with_cls(attn,0,image_input,batch_idx=idx,view=i,label=label_id) # attn_map shold be (B T H W head) 2 16 198 198 12
                else:
                    output = model(image_input, text_inputs)

                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity

            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank,
                                         timeout=datetime.timedelta(seconds=6000))
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")

    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config, args)