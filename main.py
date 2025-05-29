import os
import json
import gzip
import math
import argparse
import datetime
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from source import data_utils
from source import models
from source import losses

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, log_interval, checkpoint_dir):
    """Train the model and validate on val_loader each epoch. Save checkpoints and return best state."""
    best_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
   
    log_lines = []
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        # Training loop
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)  # forward pass
            # If using CrossEntropyLoss from torch, it expects shape (N,) targets
            target = batch.y.to(device)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            # Accumulate train metrics
            total_loss += loss.item() * batch.num_graphs
            # Compute number of correct predictions in this batch
            if out.size(-1) == 1:
                # Binary classification case (single logit). Use 0.5 threshold
                preds = (out.sigmoid() >= 0.5).long().view(-1)
            else:
                preds = out.argmax(dim=-1)  # class with highest logit
            correct += (preds.cpu() == batch.y.cpu()).sum().item()
            total_samples += batch.num_graphs
        avg_loss = total_loss / total_samples
        train_acc = correct / total_samples
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        # Validation at end of epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y.to(device)
                loss = criterion(out, target)
                val_loss += loss.item() * batch.num_graphs
                # Accuracy
                if out.size(-1) == 1:
                    preds = (out.sigmoid() >= 0.5).long().view(-1)
                else:
                    preds = out.argmax(dim=-1)
                val_correct += (preds.cpu() == batch.y.cpu()).sum().item()
                val_samples += batch.num_graphs
        avg_val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        # Checkpoint saving
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            # Save best model separately
            torch.save({'model_state': best_state}, os.path.join(checkpoint_dir, 'best_model.pth'))
        # Save periodic checkpoints (at least 5 per training)
        save_freq = max(1, epochs // 5)
        if epoch % save_freq == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pth")
            torch.save({'model_state': model.state_dict()}, ckpt_path)
        # Logging every 10 epochs
        if epoch % log_interval == 0 or epoch == epochs:
            log_line = (f"Epoch {epoch:03d}/{epochs}: "
                        f"Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, "
                        f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
            print(log_line)
            log_lines.append(log_line)
    return best_state, best_acc, history, log_lines

def main():
    parser = argparse.ArgumentParser(description="Noisy Graph Classification: Training & Inference")
    # Required arguments
    parser.add_argument('--test_path', type=str, required=True, help="Path to test dataset (JSON.gz)")
    parser.add_argument('--train_path', type=str, default=None, help="Path to training dataset (JSON.gz). If not provided, run in inference mode.")
    # Model architecture args
    parser.add_argument('--gnn_type', type=str, choices=['GCN', 'GIN'], default='GCN', help="Type of GNN to use: 'GCN' or 'GIN'")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of GNN layers")
    parser.add_argument('--emb_dim', type=int, default=64, help="Dimensionality of node/graph embeddings")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate for GNN layers")
    parser.add_argument('--virtual_node', action='store_true', help="Use virtual node to enhance graph representation")
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and inference")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay (L2 regularization strength)")
    # Noisy label handling
    parser.add_argument('--baseline_mode', type=int, choices=[0,1,2], default=0, help="Baseline mode for loss: 0=CrossEntropy, 1=LabelSmoothing, 2=NoisyCrossEntropy")
    parser.add_argument('--noise_prob', type=float, default=0.0, help="Assumed noise probability for NoisyCrossEntropyLoss (only used if baseline_mode=2)")
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('submission', exist_ok=True)
    # Create a unique folder/name for this run based on timestamp for outputs
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join('checkpoints', run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join('logs', f'train_log_{run_id}.txt')
    # Load data
    print("Loading data...")
    if args.train_path:
        # Load training data and split into train/val
        data_list = data_utils.load_graph_dataset(args.train_path, labeled=True)
        # Determine number of classes from dataset labels
        labels = [int(data.y) for data in data_list]
        num_classes = len(set(labels))
        # Stratified split (80/20)
        train_idx, val_idx = data_utils.stratified_split_indices(labels, val_ratio=0.2, random_state=42)
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        print(f"Train/Val split: {len(train_data)} training graphs, {len(val_data)} validation graphs")
    else:
        data_list = []
        train_data = []
        val_data = []
        num_classes = None  # will determine from checkpoint
    # Always load test data (labels may not be present)
    test_data = data_utils.load_graph_dataset(args.test_path, labeled=False)
    print(f"Test set: {len(test_data)} graphs")

    # Set device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model
    if args.train_path:
        # For training, we know the number of classes from labels
        in_dim = train_data[0].x.shape[1]  # dimension of node features
        model = models.GraphClassifier(in_dim, args.emb_dim, args.num_layers, args.gnn_type,
                                       num_classes, args.dropout, args.virtual_node).to(device)
    else:
        # Inference mode: load saved model config from checkpoint
        best_ckpt = torch.load(os.path.join('checkpoints', 'best_model.pth'), map_location=device)
        # If checkpoint contains model architecture info (not implemented here), we would use it.
        # For simplicity, assume user provides matching args for architecture in inference mode:
        # Determine num_classes from model weights if possible
        if 'model_state' in best_ckpt:
            state_dict = best_ckpt['model_state']
        else:
            state_dict = best_ckpt
        # Infer num_classes from final layer weight shape (works if final layer named 'linear' in our model)
        num_classes = None
        for k, v in state_dict.items():
            if k.endswith(".linear.weight"):
                num_classes = v.shape[0]
        if num_classes is None:
            raise RuntimeError("Could not infer number of classes from checkpoint. Please provide train_path or model config.")
        # Infer input feature dim similarly from first layer weight
        in_dim = None
        for k, v in state_dict.items():
            if k.startswith("convs.0"):
                # convs.0.weight for GCNConv or convs.0.nn.0.weight for GIN (depending on implementation)
                # We attempt to find in_dim from shape
                if v.dim() > 1:
                    in_dim = v.shape[1]
                    break
        if in_dim is None:
            in_dim = test_data[0].x.shape[1]  # fallback to test data features
        model = models.GraphClassifier(in_dim, args.emb_dim, args.num_layers, args.gnn_type,
                                       num_classes, args.dropout, args.virtual_node).to(device)
        model.load_state_dict(state_dict)
    # Select loss function based on baseline_mode
    if args.baseline_mode == 0:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.baseline_mode == 1:
        # Label smoothing with noise_prob (or 0.1 default if noise_prob not given)
        smooth = args.noise_prob if args.noise_prob > 0 else 0.1
        # Use PyTorch CrossEntropy with label_smoothing (available in PyTorch>=1.10)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=smooth)
    else:
        # baseline_mode 2: use custom NoisyCrossEntropyLoss
        criterion = losses.NoisyCrossEntropyLoss(noise_prob=args.noise_prob, num_classes=num_classes)

    if args.train_path:
        # DataLoaders for train and val
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        # Optimizer (Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("Starting training...")
        best_state, best_acc, history, log_lines = train(model, train_loader, val_loader, 
                                                        criterion, optimizer, device, 
                                                        epochs=args.epochs, log_interval=10, 
                                                        checkpoint_dir=checkpoint_dir)
        print(f"Training completed. Best validation accuracy = {best_acc*100:.2f}%")
        # Save final model and best model (already saved during training as best_model.pth)
        torch.save({'model_state': model.state_dict()}, os.path.join(checkpoint_dir, 'model_final.pth'))
        # Write logs to file
        with open(log_file_path, 'w') as f:
            for line in log_lines:
                f.write(line + "\n")
        # Plot training curves
        data_utils.plot_training_curves(history, save_path=os.path.join('logs', f'training_curves_{run_id}.png'))
        # Load best model weights for inference on test data
        model.load_state_dict(best_state)
    else:
        # If not training, ensure model is in eval mode
        model.eval()

    # Inference on test set
    print("Running inference on test set...")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    model.eval()
    predictions = []
    graph_ids = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            # If binary classification (num_classes == 2), we'll output 0/1
            # Otherwise, output the predicted class index
            if out.size(-1) == 1:
                # Binary case: output 0 or 1 based on sigmoid
                preds = (out.sigmoid() >= 0.5).long().view(-1).cpu().numpy()
            else:
                preds = out.argmax(dim=-1).cpu().numpy()
            # Collect predictions and IDs
            ids = batch.graph_id.cpu().numpy() if hasattr(batch, 'graph_id') else batch.batch.cpu().numpy()
            # If no explicit graph_id, batch.graph_id might not exist. In that case, use a running index.
            for i, pred in enumerate(preds):
                graph_id = int(ids[i]) if hasattr(batch, 'graph_id') else len(graph_ids)
                graph_ids.append(graph_id)
                predictions.append(int(pred))
    # Save predictions to CSV
    submission_path = os.path.join('submission', f'testset_{run_id}.csv')
    with open(submission_path, 'w') as f:
        f.write("id,label\n")
        for gid, pred in zip(graph_ids, predictions):
            f.write(f"{gid},{pred}\n")
    print(f"Saved predictions to {submission_path}")

if __name__ == "__main__":
    main()
