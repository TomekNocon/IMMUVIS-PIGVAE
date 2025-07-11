import models.components.metrics.lie_derivatives as lD


def get_equivariance_metrics(model, graph):
    errs = {
        "trans_x_deriv": lD.translation_x(model, graph),
        "trans_y_deriv": lD.translation_y(model, graph),
        "rot_deriv": lD.rotation(model, graph),
        "shear_x_deriv": lD.shear_x(model, graph),
        "shear_y_deriv": lD.shear_y(model, graph),
        "stretch_x_deriv": lD.stretch_x(model, graph),
        "stretch_y_deriv": lD.stretch_y(model, graph),
        "saturate_err": lD.saturate(model, graph),
    }
    metrics = {
        x: {
            "mean": errs[x].abs().mean().item(),
            "norm": errs[x].abs().norm(p=2).item() / errs[x].numel(),
        }
        for x in errs
    }
    flat_metrics = {f"{x}_deriv_{k}": v for x in metrics for k, v in metrics[x].items()}
    # df = pd.DataFrame.from_dict(metrics, orient="index")
    return flat_metrics
