# Initialize parameter optimizer
```python
    # training loop
    for images in image_loader:
        # images with shape [n, c, h, w]
        corrupted_images = corruption_method(images)
        loss = 0
        for _ in num_steps:
            corrupted_images = corrupted_images.detach()
            # stop gradients between inner-loop steps.
            energy_score = head(model(corrupted_images))
            # energy score with shape [n, 1]
            im_grad = autograd(energy_score.sum(), corrupted_images)
            # compute the gradient of input pixels along the direction
            # of energy maximization
            corrupted_images = corrupted_images - alpha * im_grad
            # gradient descent along the direction of energy minimization
            loss += criterion(corrupted_images, images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```