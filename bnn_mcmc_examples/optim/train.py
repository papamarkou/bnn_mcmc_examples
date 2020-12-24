def train(
    model,
    dataloader,
    optimizer,
    num_epochs,
    loss_fn=None,
    monitor_step=None,
    save_loss=False,
    save_metric=False,
    terminate_early=False,
    pred_fn=None,
    metric_fn=None,
    stop_fn=None
):
    if loss_fn is None:
        loss_fn = model.loss

    if (save_loss or save_metric) and (monitor_step is None):
        monitor_step = len(dataloader)

    loss_vals = []
    metric_vals = []
    is_break = False
    terminating_epoch = num_epochs

    for epoch in range(num_epochs):
        if terminate_early and is_break:
            terminating_epoch = epoch
            break

        for batch_idx, (input, target) in enumerate(dataloader):
            def closure():
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                return loss, output

            loss_val, output = optimizer.step(closure)

            if ((batch_idx == 0) or ((batch_idx + 1) % monitor_step == 0)):
                if save_loss:
                    loss_vals.append(loss_val.item())

                if save_metric or terminate_early:
                    metric_val = metric_fn(pred_fn(output), target)

                    if save_metric:
                        metric_vals.append(metric_val)

                    if terminate_early:
                        if stop_fn(metric_val):
                            is_break = True

    return loss_vals, metric_vals, terminating_epoch
