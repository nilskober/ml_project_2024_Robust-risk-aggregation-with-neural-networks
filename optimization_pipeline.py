from torch import optim


def optimize_model(device, model, loss_function, data_loader, loss_params, num_epochs, lr, beta1=0.99, beta2=0.995, print_every=5):
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    for epoch in range(num_epochs):
        model.train()
        for batch in data_loader:
            # Move data to device
            inputs = batch.to(device)

            # Calculate loss
            loss = loss_function(model, inputs, **loss_params)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update learning rate
        scheduler.step()

        if epoch % print_every == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')