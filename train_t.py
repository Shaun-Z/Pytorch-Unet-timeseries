import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate_1D
from unet import UNet_1D, UNet_1D_N, UNet_1D_L, UNet_1D_NN, UNet_1D_LL
from utils.data_loading import SGCCDataset
from utils.dice_score import dice_loss

# dir_data = Path('./data/attack.csv')
# dir_mask = Path('./data/label.csv')
# dir_data = Path('./data_add_noise/zx.csv')
# id = '3'
# dir_data = Path(f'./zx{id}_normalized.csv')
# dir_mask = Path(f'./zy{id}.csv')
# dir_checkpoint = Path(f'./checkpoints{id}/')

# batch_size: int = 200
batch_size: int = 200
epochs: int = 40
learning_rate: float = 1e-4 # 1e-5
save_checkpoint: bool = True
amp: bool = False
weight_decay: float = 1e-8
momentum: float = 0.999
gradient_clipping: float = 1.0,

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--val_percent', type=float, default=0.2, help='Validation percent')
    parser.add_argument('--data_name', type=str, default='SGCC', help='Name of the dataset')
    parser.add_argument('--attack_id', '-a', type=int, default=1, help='Attack ID')
    parser.add_argument('--model_name', '-n', type=str, default='UNet_1D', help='Model name')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    data_name = args.data_name
    attack_id = args.attack_id
    val_percent = args.val_percent
    model_name = args.model_name
    dir_data = Path(f'./data/{data_name}_data/data_prepared_{attack_id}/combined_dfx.csv')
    dir_mask = Path(f'./data/{data_name}_data/data_prepared_{attack_id}/combined_dfy_pseudo.csv')
    dir_checkpoint = Path(f'./checkpoints_pseudo_{data_name}/')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_classes is the number of probabilities you want to get per pixel
    if model_name == 'UNet_1D':
        model = UNet_1D(n_channels=1, n_classes=args.classes, bilinear=False)
    elif model_name == 'UNet_1D_N':
        model = UNet_1D_N(n_channels=1, n_classes=args.classes, bilinear=False)
    elif model_name == 'UNet_1D_L':
        model = UNet_1D_L(n_channels=1, n_classes=args.classes, bilinear=False)
    elif model_name == 'UNet_1D_NN':
        model = UNet_1D_NN(n_channels=1, n_classes=args.classes, bilinear=False)
    elif model_name == 'UNet_1D_LL':
        model = UNet_1D_LL(n_channels=1, n_classes=args.classes, bilinear=False)
    else:
        print('Model name is not correct')
        sys.exit(1)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    # 1. Create dataset
    try:
        dataset = SGCCDataset(dir_data, dir_mask, normalize=True)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='data') as pbar:
            for batch in train_loader:
                data, true_masks = batch['data'], batch['mask']

                assert data.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded data have {data.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                data = data.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                masks_pred = model(data) ### fake labels
                print(masks_pred.size(), true_masks.size())
                
                # prob_masks_pred = D(masks_pred)
                              
                # loss = criterion(prob_masks_pred, true_masks)### GAN structure
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 2, 1).float(),
                    multiclass=True
                )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(data.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate_1D(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        # Save the model if the validation score is the best we've seen so far.
                        if epoch == 1 or val_score > best_val_score:
                            best_val_score = val_score
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            state_dict['mask_values'] = dataset.mask_tensor
                            torch.save(state_dict, str(dir_checkpoint / 'best_checkpoint.pth'))
                            logging.info(f'Best checkpoint saved with validation score: {val_score}')

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(data[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_tensor
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')