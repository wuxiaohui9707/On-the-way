import os

loss = []

for n_res in [18, 34, 50, 101, 152]:
    # load dataset
    train, val, test = get_loaders()
    # configure model
    model_name = "ImageRegression_Resnet{}".format(n_res)
    model = get_model(model_name)
    # config trainer and fit
    trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=100, logger=logger)
    trainer.fit(model, train_loader, val_loader)

    # visualization --------------------------------------
    # vis: val
    save_path = "D:/Files/Plot"
    plot_truth_vs_prediction(
        model,
        val_loader,
        os.path.join(save_path, model_name + "val.png"),
    )
    # vis: test
    best_model_path = trainer.checkpoint_callback.best_model_path
    model = get_model(model_name, best_model_path)
    plot_truth_vs_prediction(
        model,
        test_loader,
        os.path.join(save_path, model_name + "test.png"),
    )
    loss = trainer.callback_metrics[
        "val_loss"
    ]  # find a way to extract loss from test split
    loss.append(loss)

# plot loss
plt.plot(loss)  # how the loss changes over different models


def get_model(model_name, checkpoint_path=None):
    """return the model class"""
    model_class = None
    if "18" in model_name:
        model_class = ImageRegression_Resnet18()
    elif "34" in model_name:
        model_class = ImageRegression_Resnet34()
    if checkpoint_path is not None:
        model_class.load_state_dict(torch.load(checkpoint_path))
    return model_class


def get_loaders(batch_size=32):
    transform = get_transform()
    raw_datasets = load_dataset(
        "Niche-Squad/mock-dots",
        "regression-one-class",
        download_mode="force_redownload",
    )
    train_loader, val_loader, test_loader = get_loaders(raw_datasets, transform, 32)
    return train_loader, val_loader, test_loader
