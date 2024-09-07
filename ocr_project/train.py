import os
import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.text import CharErrorRate

from tqdm import tqdm

from ocr_project.dataset import get_dataloaders
from ocr_project.models.baseline_model import BaselineModel
from ocr_project.models.branched_model import BranchedModel
from ocr_project.utils import get_device, calculate_accuracy, calculate_cer


def Train(model, dataloader, ctc_loss, optimizer, device='cpu'):
    model.train()
    cer_metric = CharErrorRate()
    index_to_token = dataloader.dataset.dataset.index_to_token

    total_loss = 0
    all_predictions = []
    all_targets = []

    for inputs, targets, target_lengths in tqdm(dataloader, desc='Train_Batches'):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        y_pred = model(inputs)
        
        # Calculate CTC Loss
        input_lengths = torch.full(size=(inputs.size(0),), fill_value=y_pred.size(1), dtype=torch.long)
        loss = ctc_loss(y_pred.permute(1,0,2), targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Calculate CER
        predictions = y_pred.argmax(2)
        calculate_cer(cer_metric, predictions.cpu().numpy(), targets.cpu().numpy(), index_to_token)
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(dataloader)
    cer = cer_metric.compute().item()*100
    cer_metric.reset()

    accuracy = calculate_accuracy(all_predictions, all_targets, index_to_token)
    print(f'Train_Loss: {epoch_loss:.4f}, Train_CER: {cer:.2f}%, Train_Accuracy: {accuracy:.2f}%')
    
    return model


def Test(model, dataloader, ctc_loss, device='cpu'):
    model.eval()
    num_batches = len(dataloader)
    cer_metric = CharErrorRate()
    index_to_token = dataloader.dataset.dataset.index_to_token

    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, target_lengths in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            y_pred = model(inputs)
            
            # Calculate CTC Loss
            input_lengths = torch.full(size=(inputs.size(0),), fill_value=y_pred.size(1), dtype=torch.long)
            loss = ctc_loss(y_pred.permute(1, 0, 2), targets, input_lengths, target_lengths)

            total_loss += loss.item()
            predictions = y_pred.argmax(2)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            calculate_cer(cer_metric, predictions.cpu().numpy(), targets.cpu().numpy(), index_to_token)
    
            del inputs
            del targets
            del target_lengths

            if device == 'cuda':
              torch.cuda.empty_cache()
    
    epoch_loss = total_loss / num_batches
    cer = cer_metric.compute().item()*100
    cer_metric.reset()

    accuracy = calculate_accuracy(all_predictions, all_targets, index_to_token)
    print(f'Test_Loss: {epoch_loss:.4f}, Test_CER: {cer:.2f}%, Test_Accuracy: {accuracy:.2f}%')

    return model


def save_checkpoint(model, optimizer, epoch, save_to):
    checkpoint_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint_data, os.path.join(save_to, f'cp_{epoch}'))


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint_data = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint_data["model"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    start_epoch = checkpoint_data["epoch"] + 1

    return start_epoch


def main(args):
    device = get_device()
    print(f"[INFO] Running on device: {device}")

    print(f"[INFO] Loading data")
    train_dataloader, test_dataloader = get_dataloaders(
        args.root_dir,
        args.annotation_file,
        args.token_file,
        args.batch_size
    )

    print(f"[INFO] Creating {args.model_type} model")
    if args.model_type == 'baseline':
        model = BaselineModel(num_classes=77).to(device)
    elif args.model_type == 'branched':
        model = BranchedModel(num_classes=77).to(device)
    else:
        raise ValueError(
            f'Model type "{args.model_type}" is not supported. '
            'Supported types are: ["baseline", "branched"].'
        )

    ctc_loss = torch.nn.CTCLoss(blank=0)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0

    # load checkpoint if given
    if args.checkpoint is not None:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint)
        print(f"[INFO] Successfully loaded checkpoint: {args.checkpoint}")

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=start_epoch-1)

    # Training Loop
    print(f"[INFO] Start training")
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch [{epoch}/{args.epochs - 1}]:')

        Train(model, train_dataloader, ctc_loss, optimizer, device)

        scheduler.step()
        save_checkpoint(model, optimizer, epoch, args.checkpoints_dir)

        Test(model, test_dataloader, ctc_loss, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR Model")

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for dataloaders')
    parser.add_argument('--step_size', type=int, default=1, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.9, help='Scheduler learning rate decay factor')

    # Dataset parameters
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for dataset')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to annotations file')
    parser.add_argument('--token_file', type=str, required=True, help='Path to tokens file')
    parser.add_argument('--checkpoints_dir', type=str, required=True, help='Path to directory in which checkpoints will be saved')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to checkpoint to load')
    parser.add_argument('--model_type', type=str, required=True, help='Model type. Supported types are: ["baseline", "branched"].')

    args = parser.parse_args()

    main(args)
