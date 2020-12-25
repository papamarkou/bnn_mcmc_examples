def train(
    model,
    dataloader,
    optimizer,
    num_epochs,
    loss_fn=None,
    monitor_step=None,
    save_loss=False,
    save_metric=False,
    save_metric_mean=False,
    terminate_early=False,
    pred_fn=None,
    metric_fn=None,
    stop_fn=None
):
    if loss_fn is None:
        loss_fn = model.loss

    if (save_loss or save_metric or save_metric_mean) and (monitor_step is None):
        monitor_step = len(dataloader)

    loss_vals = []
    metric_vals = []
    metric_mean_vals = []
    terminating_epoch = num_epochs
    num_batches = 0
    metric_mean_val = 0
    is_break = False

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

            num_batches = num_batches + 1

            loss_val, output = optimizer.step(closure)

            if ((batch_idx == 0) or ((batch_idx + 1) % monitor_step == 0)):
                if save_loss:
                    loss_vals.append(loss_val.item())

                if save_metric or save_metric_mean or terminate_early:
                    metric_val = metric_fn(pred_fn(output), target)

                    if save_metric:
                        metric_vals.append(metric_val)

                    if save_metric_mean:
                        metric_mean_val = (metric_mean_val * (num_batches - 1) + metric_val) / num_batches
                        metric_mean_vals.append(metric_mean_val)

                    if terminate_early:
                        if stop_fn(metric_val):
                            is_break = True

    return {
        'loss_vals': loss_vals,
        'metric_vals': metric_vals,
        'metric_mean_vals': metric_mean_vals,
        'terminating_epoch': terminating_epoch,
        'num_batches': num_batches
    }
