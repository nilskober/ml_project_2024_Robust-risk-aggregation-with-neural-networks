import torch
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


def optimize_model(device, model, additional_parameters, loss_function, data_loader, loss_params, num_epochs_total, start_decay_at, lr, lr_lambda = 0.1, beta1=0.99, beta2=0.995,
                   print_every=5, test_every=50, test_data_loader=None):
    '''
    additional_parameters: list of dictionary of additional parameters for the loss function, e.g. lambda. Assume the following strucutre:
    additional_parameters = [
        {
            'name': 'lambda',
            'param': torch.tensor([1.0], requires_grad=True),
            'update_every': 50,
            'start_optimizing_at': 0,
            'initial_lr': 0.1,
            'start_decay_at': 0,
            'decay_every': 50,
            'decay_rate': 0.98,
            'start_decay_at': 0
            'lower_bound': 0.0,
        },
        ...
    ]
    '''
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    optimizers_additional = [
        optim.SGD([p['param']], lr=p['initial_lr']) for p in additional_parameters
    ]
    # Learning rate scheduler
    scheduler = CustomExponentialLR(optimizer, gamma=0.98, step_size=50, start_decay_at=start_decay_at)
    scheduler_additional = [
        CustomExponentialLR(opt, gamma=p['decay_rate'], step_size=p['decay_every'], start_decay_at=p['start_decay_at'])
        for p, opt in zip(additional_parameters, optimizers_additional)
    ]

    loss_trajectory_train = []
    loss_trajectory_test = []

    for epoch in range(num_epochs_total):
        model.train()
        batch = next(data_loader)
        # Move data to device
        inputs = batch.to(device)
        outputs = model(inputs)

        additional_parameters_values = {p['name']: p['param'] for p in additional_parameters}
        # Calculate loss
        loss = loss_function(inputs, outputs, additional_parameters_values, **loss_params)
        loss_trajectory_train.append((epoch, loss.item()))


        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update additional parameters
        for p, opt, scheduler_ in zip(additional_parameters, optimizers_additional, scheduler_additional):
            if epoch == p['start_optimizing_at']:
                opt.zero_grad()
            if epoch > p['start_optimizing_at'] and (epoch - p['start_optimizing_at']) % p['update_every'] == 0:
                # normalize the gradient
                p['param'].grad /= p['update_every']
                opt.step()
                # clamp the parameter if necessary
                if p['lower_bound'] is not None:
                    with torch.no_grad():
                        p['param'].clamp(min=p['lower_bound'])
                opt.zero_grad()
            scheduler_.step()

        scheduler.step()

        if epoch % print_every == 0:
            print(f'Epoch [{epoch}/{num_epochs_total}], Loss: {loss.item():.4f}')
            print('Additional parameters:')
            for p in additional_parameters:
                print(f'{p["name"]}: {p["param"].item()}')

        if (epoch % test_every == 0 or epoch == num_epochs_total-1) and test_data_loader is not None:
            model.eval()
            batch = next(test_data_loader)
            inputs = batch.to(device)
            outputs = model(inputs)
            loss = loss_function(inputs, outputs, additional_parameters_values, **loss_params)
            loss_trajectory_test.append((epoch, loss.item()))
            print(f'Epoch [{epoch}/{num_epochs_total}], Test Loss: {loss.item():.4f}')

    return {
        'loss_trajectory_train': loss_trajectory_train,
        'loss_trajectory_test': loss_trajectory_test
    }



