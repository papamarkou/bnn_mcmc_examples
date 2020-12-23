def train(model, dataloader, optimizer, num_epochs, loss_fn=None, save_loss=False, loss_save_step=None):
    if loss_fn is None:
        loss_fn = model.loss
    
    if save_loss and (loss_save_step is None):
        loss_save_step = len(dataloader)
    
    loss_vals = []

    for epoch in range(num_epochs):
        for batch_idx, (input, target) in enumerate(dataloader):
            def closure():
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                return loss

            loss_val = optimizer.step(closure)

            if save_loss and ((batch_idx == 0) or ((batch_idx + 1) % loss_save_step == 0)):
                loss_vals.append(loss_val.item())

    return loss_vals
