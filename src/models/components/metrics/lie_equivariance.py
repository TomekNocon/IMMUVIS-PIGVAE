import torch
import pandas as pd
import lie_derivatives as lD
import torch.nn.functional as F

def get_equivariance_metrics(model, minibatch):
    x, y = minibatch
    if torch.cuda.is_available():
        model = model.cuda()
        x, y = x.cuda(), y.cuda()

    model = model.eval()

    model_probs = lambda x: F.softmax(model(x), dim=-1)

    errs = {
        "trans_x_deriv": lD.translation_lie_deriv(model_probs, x, axis="x"),
        "trans_y_deriv": lD.translation_lie_deriv(model_probs, x, axis="y"),
        "rot_deriv": lD.rotation_lie_deriv(model_probs, x),
        "shear_x_deriv": lD.shear_lie_deriv(model_probs, x, axis="x"),
        "shear_y_deriv": lD.shear_lie_deriv(model_probs, x, axis="y"),
        "stretch_x_deriv": lD.stretch_lie_deriv(model_probs, x, axis="x"),
        "stretch_y_deriv": lD.stretch_lie_deriv(model_probs, x, axis="y"),
        "saturate_err": lD.saturate_lie_deriv(model_probs, x),
    }
    
    metrics = {x: pd.Series(errs[x].abs().cpu().data.numpy().mean(-1)) for x in errs}
    df = pd.DataFrame.from_dict(metrics)
    return df