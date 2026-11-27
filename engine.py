import torch.nn as nn
import torch


def train_model(train_dataloader : torch.utils.data.dataloader.DataLoader,
                model : nn.Module,
                loss_function : nn.Module,
                optimizer : torch.optim.Optimizer,
                device : torch.device,
                ):
    train_loss = 0
    train_acc = 0
    running_pixels = 0

    model = model.to(device)
    model.train()
    for batch_idx, (image,mask,weight) in enumerate(train_dataloader):

        image = image.to(device)
        mask = mask.to(device)
        weight = weight.to(device)

        prediction_mask = model(image)
        loss_map = loss_function(prediction_mask,mask)
        loss = (loss_map * weight).mean()

        preds = prediction_mask.argmax(dim=1)
        acc = (preds == mask).sum().item()
        running_pixels += mask.numel()

        train_loss += loss.item()
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0 and batch_idx != 0:
            print(f"Batch : {batch_idx}")



    train_loss /= len(train_dataloader)
    train_acc /= running_pixels

    return train_loss,train_acc


def validation_model(val_dataloader : torch.utils.data.dataloader.DataLoader,
                model : nn.Module,
                loss_function : nn.Module,
                device : torch.device,
                    ):
    val_loss = 0
    val_acc = 0
    running_pixels = 0

    model = model.to(device)
    model.train()
    for batch_idx, (image, mask, weight) in enumerate(val_dataloader):

        image = image.to(device)
        mask = mask.to(device)
        weight = weight.to(device)

        prediction_mask = model(image)
        loss_map = loss_function(prediction_mask, mask)
        loss = (loss_map * weight).mean()

        preds = prediction_mask.argmax(dim=1)
        acc = (preds == mask).sum().item()
        running_pixels += mask.numel()

        val_loss += loss.item()
        val_acc += acc


        if batch_idx % 100 == 0 and batch_idx != 0:
            print(f"Batch : {batch_idx}")

    val_loss /= len(val_dataloader)
    val_acc /= running_pixels

    return val_loss, val_acc




