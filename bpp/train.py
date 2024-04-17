import os
import torch
import torch.nn as nn
import argparse
import json
import shutil
import glob
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from model import RNNBinPacking
from data_processing import load_data
from utils import count_parameters, combined_loss, load_checkpoint

torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a deep learning model.')
    parser.add_argument('--dim', type=int, default=16, help='Dimmension of model')
    parser.add_argument('--head', type=int, default=4, help='Number of heads')
    parser.add_argument('--transformer_layers', type=int, default=1, help='Number of Transformer layers')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--seed', type=int, default=1234, help='torch.manual_seed')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--dataset', nargs='+', default=['bpp_dataset'], help='Datasets')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu', type=bool, default=True, help='Usage of GPU')
    parser.add_argument('--checkpoint', type=str, default="", help='Folder of checkpoint')
    parser.add_argument('--multiply_a', type=int, default=1, help='Multiply of permutation operator')

    args = parser.parse_args()
    hidden_size = args.dim
    nhead = args.head
    num_transformer_layers = args.transformer_layers
    num_rnn_layers = args.rnn_layers
    num_fc_neurons = hidden_size
    d_ff = 4 * hidden_size
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    dataset = args.dataset
    learning_rate = args.learning_rate
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    multiply_a = args.multiply_a


    torch.manual_seed(seed)

    # Logging
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    root_dir = f'./{current_time}'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    logging.basicConfig(filename=os.path.join(root_dir, 'training_log.log'), level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Model
    train_loader, val_loader, test_loader, pos_weight = load_data(dataset, train_batch_size=batch_size)
    model = RNNBinPacking(hidden_size, nhead, num_transformer_layers, num_rnn_layers, num_fc_neurons, d_ff).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_params, trainable_params = count_parameters(model)
    logging.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # Store file
    plot_dir = os.path.join(root_dir, 'plots')
    model_dir = os.path.join(root_dir, 'models')
    code_dir = os.path.join(root_dir, 'code')
    checkpoint_path = f'{args.checkpoint}/models/best_model_checkpoint.pth'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)  
    args_dict = vars(args)
    with open(os.path.join(root_dir, 'args_info.json'), 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=4)
    src_files = glob.glob('./*.py')
    for file in src_files:
        shutil.copy(file, code_dir)
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, train_losses, train_accuracies, val_losses, val_accuracies, val_tprs, val_tnrs, val_fprs, val_fnrs, num_epochs, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)
    else:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        val_tprs, val_tnrs, val_fprs, val_fnrs = [], [], [], []
        start_epoch = 0
        num_epochs = epochs
        best_val_loss = float('inf')

    # Train
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            loss, outputs = combined_loss(batch_data, batch_labels, model, criterion, multiply_a=multiply_a)
            train_loss += loss.item()

            predicted = (outputs.sigmoid() > 0.5).float()
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                loss, outputs = combined_loss(batch_data, batch_labels, model, criterion)
                val_loss += loss.item()

                predicted = (outputs.sigmoid() > 0.5).float()
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
                
                true_positives += ((predicted == 1) & (batch_labels == 1)).sum().item()
                true_negatives += ((predicted == 0) & (batch_labels == 0)).sum().item()
                false_positives += ((predicted == 1) & (batch_labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (batch_labels == 1)).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (true_negatives + false_positives)
        tnr = true_negatives / (true_negatives + false_positives)
        fnr = false_negatives / (true_positives + false_negatives)

        val_tprs.append(tpr)
        val_tnrs.append(tnr)
        val_fprs.append(fpr)
        val_fnrs.append(fnr)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc:.2f}%, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Plotting metrics after each epoch
        # Plotting Loss
        plt.figure(figsize=(7, 5))
        plt.plot(range(epoch + 1), train_losses, label='Training Loss')
        plt.plot(range(epoch + 1), val_losses, label='Validation Loss')
        plt.legend()
        plt.title(f"Loss vs. Epochs (Up to Epoch {epoch + 1})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"{plot_dir}/loss_epoch.png")
        plt.close()
        
        # Plotting Accuracy
        plt.figure(figsize=(7, 5))
        plt.plot(range(epoch + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(epoch + 1), val_accuracies, label='Validation Accuracy')
        plt.legend()
        plt.title(f"Accuracy vs. Epochs (Up to Epoch {epoch + 1})")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.savefig(f"{plot_dir}/accuracy_epoch.png")
        plt.close()

        # Plotting Validation's TPR and FPR
        plt.figure(figsize=(7, 5))
        plt.plot(range(epoch + 1), val_tprs, label='True Positive Rate (Validation)')
        plt.plot(range(epoch + 1), val_fprs, label='False Positive Rate (Validation)')
        plt.plot(range(epoch + 1), val_tnrs, label='True Negative Rate (Validation)')
        plt.plot(range(epoch + 1), val_fnrs, label='False Negative Rate (Validation)')
        plt.legend()
        plt.title(f"Validation TPR & FPR & TNR & FNR vs. Epochs (Up to Epoch {epoch + 1})")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.grid(True)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.savefig(f"{plot_dir}/tpr_fpr_tnr_fnr_epoch.png")
        plt.close()


        # Save model checkpoint with more info
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'val_tprs': val_tprs,
                'val_tnrs': val_tnrs,
                'val_fprs': val_fprs,
                'val_fnrs': val_fnrs,
                'num_epochs': num_epochs,
                'best_val_loss': best_val_loss
            }, f'{model_dir}/model_epoch_{epoch}.pth')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'val_tprs': val_tprs,
                'val_tnrs': val_tnrs,
                'val_fprs': val_fprs,
                'val_fnrs': val_fnrs,
                'num_epochs': num_epochs,
                'best_val_loss': best_val_loss
            }, f'{model_dir}/best_model_checkpoint.pth')

    logging.info("Training Finished.")