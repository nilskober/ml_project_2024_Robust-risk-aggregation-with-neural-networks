from torch import optim
from torch.optim.lr_scheduler import ExponentialLR


class CustomExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, step_size=50, start_decay_at=0, last_epoch=-1):
        self.step_size = step_size
        self.start_decay_at = start_decay_at
        super(CustomExponentialLR, self).__init__(optimizer, gamma, last_epoch)

    def step(self, epoch=None):
        if self.start_decay_at <= self.last_epoch and (self.last_epoch + 1) % self.step_size == 0:
            return super().step(epoch)
        else:
            self.last_epoch += 1

def optimize_model(device, model, loss_function, data_loader, loss_params, num_epochs_total, start_decay_at, lr, lr_lambda = 0.1, beta1=0.99, beta2=0.995,
                   print_every=5):
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    lambda_optimizer = optim.Adam([model.lambda_par], lr=lr_lambda, betas=(beta1, beta2))
    # Learning rate scheduler
    scheduler = CustomExponentialLR(optimizer, gamma=0.98, step_size=50, start_decay_at=start_decay_at)
    lambda_sheduler = CustomExponentialLR(lambda_optimizer, gamma=0.98, step_size=50, start_decay_at=start_decay_at)

    for epoch in range(num_epochs_total):
        model.train()
        batch = next(data_loader)
        # Move data to device
        inputs = batch.to(device)
        outputs = model(inputs)

        # Calculate loss
        loss = loss_function(inputs, outputs, **loss_params)

        # Backward pass and optimization
        optimizer.zero_grad()
        lambda_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lambda_optimizer.step()
        # TODO: optimize lambda separately as in the paper



        # Update learning rate
        if epoch % 50 == 0:
            scheduler.step()
            lambda_sheduler.step()

        if epoch % print_every == 0:
            print(f'Epoch [{epoch}/{num_epochs_total}], Loss: {loss.item():.4f}')



