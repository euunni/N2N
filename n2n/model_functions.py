import torch
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


def check_available_device() -> str:
    """Return the device available to torch.

    Returns
    -------
    device : str
        The device available to torch.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(model, optimizer, criterion, data_loader, device):
    model.train()
    running = 0.0
    iterator = data_loader
    total = len(data_loader)
    if _tqdm is not None:
        try:
            iterator = _tqdm(data_loader, total=total, desc="Train", leave=False)
        except Exception:
            iterator = data_loader
    for step, (inputs, targets) in enumerate(iterator, 1):
        inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running += loss.item()
        if iterator is not data_loader and hasattr(iterator, "set_postfix"):
            try:
                iterator.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running/step:.4f}")
            except Exception:
                pass
    return running / total if total else 0.0


def validate(model, criterion, data_loader, device):
    model.eval()
    running = 0.0
    iterator = data_loader
    total = len(data_loader)
    if _tqdm is not None:
        try:
            iterator = _tqdm(data_loader, total=total, desc="Val", leave=False)
        except Exception:
            iterator = data_loader
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(iterator, 1):
            inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running += loss.item()
            if iterator is not data_loader and hasattr(iterator, "set_postfix"):
                try:
                    iterator.set_postfix(avg=f"{running/step:.4f}")
                except Exception:
                    pass
    return running / total if total else 0.0


def predict(model, data_loader, device, show_progress: bool = False, desc: str = "Predict"):
    model.eval()
    predictions = []
    iterator = data_loader
    if show_progress and _tqdm is not None:
        try:
            iterator = _tqdm(data_loader, total=len(data_loader), desc=desc)
        except Exception:
            iterator = data_loader
    with torch.no_grad():
        for inputs in iterator:
            inputs = inputs[0].unsqueeze(1)
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
    predictions = torch.cat(predictions, dim=0).squeeze(1)
    return predictions
