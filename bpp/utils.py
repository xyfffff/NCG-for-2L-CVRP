import torch


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def shuffle_operator(tensor, multiply=1):
    result = []
    
    for i in range(tensor.size(0)): # (batch, n, 3)
        sequence = tensor[i]    # (n, 3)
        
        padding_mask = sequence[:, 2] == -1 # (n,)
        non_padding = sequence[~padding_mask]   # (k, 3)
        non_padding = non_padding[None, :, :].repeat(multiply, 1, 1)    # (multiply, k, 3)
        padding = sequence[padding_mask]    # (n-k, 3)
        padding = padding[None, :, :].repeat(multiply, 1, 1)    #  (multiply, n-k, 3)
        
        rand_nums = torch.rand(multiply, non_padding.size(1), device=tensor.device) # (multiply, k)
        
        _, indices = torch.sort(non_padding[:, :, 2] + rand_nums)   # (multiply, k)
        expanded_indices = indices[:, :, None].expand_as(non_padding)   # (multiply, k, 3)
        shuffled_sequence = torch.cat([non_padding.gather(index=expanded_indices, dim=1), padding], dim=1)   # (multiply, n, 3)
    
        result.append(shuffled_sequence.unsqueeze(0))   # (1, multiply, n, 3)
    return torch.cat(result, dim=0) # (batch, multiply, n, 3)


def combined_loss(x, y, model, criterion, multiply_a=1):

    if multiply_a == 1:
        y_pred = model(x)
        bce_loss = criterion(y_pred, y)
        return bce_loss, y_pred

    elif multiply_a != 1:
        x_a = torch.cat([x[:, None, :, :], shuffle_operator(x, multiply_a-1)], dim=1)  # (batch, a, n, 3)
        batch, _, n, _ = x_a.size()
        y_pred = model(x_a.view(batch*multiply_a, n, 3))    # (batch*a, 1)

        bce_loss = criterion(y_pred, y.repeat_interleave(repeats=multiply_a, dim=0))

        y_pred = y_pred.view(batch, multiply_a)[:, 0:1]

        return bce_loss, y_pred
    
    else:
        raise NotImplementedError


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    train_accuracies = checkpoint['train_accuracies']
    val_losses = checkpoint['val_losses']
    val_accuracies = checkpoint['val_accuracies']
    val_tprs = checkpoint['val_tprs']
    val_tnrs = checkpoint['val_tnrs']
    val_fprs = checkpoint['val_fprs']
    val_fnrs = checkpoint['val_fnrs']
    num_epochs = checkpoint['num_epochs']
    best_val_loss = checkpoint['best_val_loss']
    
    return model, optimizer, start_epoch, train_losses, train_accuracies, val_losses, val_accuracies, val_tprs, val_tnrs, val_fprs, val_fnrs, num_epochs, best_val_loss
