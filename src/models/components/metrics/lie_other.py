import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

import lie_transforms as lT

# Needs TBC since I need a graph input of this so will see
def translation_sample_invariance(
        model,
        img,
        model_out,
        axis='x',
        repetitions: int = 10,
        eta=2.0
):
    """ Lie derivative of model with respect to translation vector, assumes scalar output """
    shifted_model = lambda t: model(lT.translate(img, t, axis))
    errors = []
    for _ in range(repetitions):
        t_sample = (2 * eta) * torch.rand(1) - eta
        mse = F.mse_loss(model_out, shifted_model(t_sample))
        errors.append(mse)
    return torch.stack(errors).mean(0).unsqueeze(0)

def rotation_sample_invariance(
        model,
        img,
        model_out,
        repetitions: int = 10,
        eta=np.pi//16
):
    """ Lie derivative of model with respect to rotation, assumes scalar output """
    rotated_model = lambda theta: model(lT.rotate(img, theta))
    errors = []
    for _ in range(repetitions):
        theta_sample = (2 * eta) * torch.rand(1) - eta
        mse = F.mse_loss(model_out, rotated_model(theta_sample))
        errors.append(mse)
    return torch.stack(errors).mean(0).unsqueeze(0)

"""TODO: read what this does """
def get_equivariance_metrics(model, minibatch, num_probes=20):
    x, y = minibatch
    if torch.cuda.is_available():
        model = model.cuda()
        x, y = x.cuda(), y.cuda()

    model = model.eval()

    model_probs = lambda x: F.softmax(model(x), dim=-1)
    model_out = model_probs(x)

    yhat = model_out.argmax(dim=1)  # .cpu()
    acc = (yhat == y).cpu().float().data.numpy()

    metrics = {}
    metrics["acc"] = pd.Series(acc)

    with torch.no_grad():
        for shift_x in range(8):
            rolled_img = torch.roll(x, shift_x, 2)
            rolled_yhat = model(rolled_img).argmax(dim=1)
            consistency = (rolled_yhat == yhat).cpu().data.numpy()
            metrics["consistency_x" + str(shift_x)] = pd.Series(consistency)
        for shift_y in range(8):
            rolled_img = torch.roll(x, shift_y, 3)
            rolled_yhat = model(rolled_img).argmax(dim=1)
            consistency = (rolled_yhat == yhat).cpu().data.numpy()
            metrics["consistency_y" + str(shift_y)] = pd.Series(consistency)

        
        metrics['trans_x_sample'] = translation_sample_invariance(model_probs,x,model_out,axis='x').abs().cpu().data.numpy()
        metrics['trans_y_sample'] = translation_sample_invariance(model_probs,x,model_out,axis='y').abs().cpu().data.numpy()
        metrics['rotate_sample'] = rotation_sample_invariance(model_probs,x,model_out).abs().cpu().data.numpy()

    df = pd.DataFrame.from_dict(metrics)
    return df