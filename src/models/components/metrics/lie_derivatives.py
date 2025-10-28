from collections.abc import Callable
from functools import wraps
from typing import Literal

import torch

import models.components.metrics.lie_transforms as lT


def jvp(f, x, u):
    """Jacobian vector product Df(x)u vs typical autograd VJP vTDF(x).

    Uses two backwards passes: computes (vTDF(x))u and then derivative wrt to v to get DF(x)u
    """
    with torch.enable_grad():
        y = f(x)
        v = torch.ones_like(
            y,
            requires_grad=True,
        )  # Dummy variable (could take any value)
        vj = torch.autograd.grad(y, [x], [v], create_graph=True)
        ju = torch.autograd.grad(vj, [v], [u], create_graph=True)
        return ju[0]


def lie_derivative(
    transform_type: Literal[
        "translate",
        "rotate",
        "hyperbolic_rotation",
        "scale",
        "shear",
        "stretch",
        "saturate",
    ],
    *,
    patch_size: int = 4,
    axis: str = "x",
    return_scalar: bool = False,
):
    def decorator(transform_fn: Callable):
        @wraps(transform_fn)
        def wrapper(model: Callable, graph):
            batch_size = graph.node_features.shape[0]
            inp_imgs = lT.restore_tensor(
                graph.node_features,
                batch_size,
                1,
                24,
                24,
                patch_size,
            )

            if not lT.img_like(inp_imgs.shape):
                return torch.tensor(0.0, device=inp_imgs.device)

            def transformed_model(t: torch.Tensor) -> torch.Tensor:
                transformed_img = (
                    transform_fn(inp_imgs, t, axis)
                    if transform_type in ["translate", "shear", "stretch"]
                    else transform_fn(inp_imgs, t)
                )

                transformed_graph = lT.patch_tensor(transformed_img, patch_size)
                graph.node_features = transformed_graph

                _, z, *_ = model(graph, training=False, tau=1.0)
                z_tensor = lT.restore_tensor(
                    z.node_features,
                    batch_size,
                    1,
                    24,
                    24,
                    patch_size,
                )

                if lT.img_like(z_tensor.shape):
                    if transform_type in ["translate", "shear", "stretch"]:
                        z_tensor = transform_fn(z_tensor, -t, axis)
                    elif transform_type == "rotate":
                        z_tensor = transform_fn(z_tensor, -t)

                return z_tensor

            t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
            lie_deriv = jvp(transformed_model, t, torch.ones_like(t))
            return lie_deriv.norm() if return_scalar else lie_deriv

        return wrapper

    return decorator


@lie_derivative("translate", axis="x", patch_size=4, return_scalar=False)
def translation_x(imgs, t, axis):
    return lT.translate(imgs, t, axis)


@lie_derivative("translate", axis="y", patch_size=4, return_scalar=False)
def translation_y(imgs, t, axis):
    return lT.translate(imgs, t, axis)


@lie_derivative("rotate", patch_size=4, return_scalar=False)
def rotation(imgs, t):
    return lT.rotate(imgs, t)


@lie_derivative("hyperbolic_rotation", patch_size=4, return_scalar=False)
def hyperbolic_rotation(imgs, t):
    return lT.hyperbolic_rotate(imgs, t)


@lie_derivative("scale", patch_size=4, return_scalar=False)
def scale(imgs, t):
    return lT.scale(imgs, t)


@lie_derivative("shear", axis="x", patch_size=4, return_scalar=False)
def shear_x(imgs, t, axis):
    return lT.shear(imgs, t, axis)


@lie_derivative("shear", axis="y", patch_size=4, return_scalar=False)
def shear_y(imgs, t, axis):
    return lT.shear(imgs, t, axis)


@lie_derivative("stretch", axis="x", patch_size=4, return_scalar=False)
def stretch_x(imgs, t, axis):
    return lT.stretch(imgs, t, axis)


@lie_derivative("stretch", axis="y", patch_size=4, return_scalar=False)
def stretch_y(imgs, t, axis):
    return lT.stretch(imgs, t, axis)


@lie_derivative("saturate", patch_size=4, return_scalar=False)
def saturate(imgs, t):
    return lT.saturate(imgs, t)


# def translation_lie_deriv(
#     model: Callable,
#     graph,
#     axis: str = "x",
#     patch_size: int = 4,
#     return_scalar: bool = False
# ) -> torch.Tensor:
#     """Lie derivative of model with respect to translation vector, output can be a scalar or an image"""
#     # vector = vector.to(inp_imgs.device)
#     batch_size = graph.node_features.shape[0]
#     inp_imgs = lT.restore_tensor(graph.node_features, batch_size, 1, 24, 24, patch_size)
#     if not lT.img_like(inp_imgs.shape):
#         return torch.tensor(0.0, device=inp_imgs.device)

#     def shifted_model(t):
#         # print("Input shape",inp_imgs.shape)
#         shifted_img = lT.translate(inp_imgs, t, axis)
#         shifted_graph = lT.patch_tensor(shifted_img, patch_size)
#         new_graph = graph.clone()
#         new_graph.node_features = shifted_graph
#         _, z, _, _, _, _  = model(new_graph, training=False, tau=1.0)
#         z_restore = lT.restore_tensor(z.node_features, batch_size, 1, 24, 24, patch_size)

#         if lT.img_like(z_restore.shape):
#             z_restore = lT.translate(z_restore, -t, axis)
#         # print('zshape',z.shape)
#         return z_restore

#     t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
#     lie_deriv = jvp(shifted_model, t, torch.ones_like(t, requires_grad=True))

#     if return_scalar:
#         return lie_deriv.norm()

#     return lie_deriv


# def rotation_lie_deriv(model, graph, patch_size: int = 4):
#     """Lie derivative of model with respect to rotation, assumes scalar
#     output"""
#     batch_size = graph.node_features.shape[0]
#     inp_imgs = lT.restore_tensor(graph.node_features, batch_size, 1, 24, 24, patch_size)
#     if not lT.img_like(inp_imgs.shape):
#         return 0.0

#     def rotated_model(t):
#         rotated_img = lT.rotate(inp_imgs, t)
#         rotated_graph = lT.patch_tensor(rotated_img, patch_size)
#         graph.node_features = rotated_graph
#         _, z, _, _, _, _  = model(graph, training=False, tau=1.0)
#         z_restore = lT.restore_tensor(z.node_features, batch_size, 1, 24, 24, 4)
#         if lT.img_like(z_restore.shape):
#             z_restore = lT.rotate(z_restore, -t)
#         return z_restore

#     t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
#     lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
#     return lie_deriv


# def hyperbolic_rotation_lie_deriv(model, graph, patch_size: int = 4):
#     """Lie derivative of model with respect to rotation, assumes scalar
#     output"""
#     batch_size = graph.node_features.shape[0]
#     inp_imgs = lT.restore_tensor(graph.node_features, batch_size, 1, 24, 24, patch_size)
#     if not lT.img_like(inp_imgs.shape):
#         return 0.0

#     def rotated_model(t):
#         rotated_img = lT.hyperbolic_rotate(inp_imgs, t)
#         rotated_graph = lT.patch_tensor(rotated_img, patch_size)
#         graph.node_features = rotated_graph
#         _, z, _, _, _, _  = model(graph, training=False, tau=1.0)
#         z_restore = lT.restore_tensor(z.node_features, batch_size, 1, 24, 24, 4)
#         if lT.img_like(z_restore.shape):
#             z_restore = lT.hyperbolic_rotate(z_restore, -t)
#         return z_restore

#     t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
#     lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
#     return lie_deriv


# def scale_lie_deriv(model, graph, patch_size: int = 4):
#     """Lie derivative of model with respect to rotation, assumes scalar
#     output"""
#     batch_size = graph.node_features.shape[0]
#     inp_imgs = lT.restore_tensor(graph.node_features, batch_size, 1, 24, 24, patch_size)
#     if not lT.img_like(inp_imgs.shape):
#         return 0.0

#     def scaled_model(t):
#         scaled_img = lT.scale(inp_imgs, t)
#         scaled_graph = lT.patch_tensor(scaled_img, patch_size)
#         graph.node_features = scaled_graph
#         _, z, _, _, _, _  = model(graph, training=False, tau=1.0)
#         z_restore = lT.restore_tensor(z.node_features, batch_size, 1, 24, 24, 4)
#         if lT.img_like(z_restore.shape):
#             z_restore = lT.scale(z_restore, -t)
#         return z_restore

#     t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
#     lie_deriv = jvp(scaled_model, t, torch.ones_like(t))
#     return lie_deriv


# def shear_lie_deriv(model, graph, axis="x", patch_size: int = 4):
#     """Lie derivative of model with respect to shear, assumes scalar output"""
#     batch_size = graph.node_features.shape[0]
#     inp_imgs = lT.restore_tensor(graph.node_features, batch_size, 1, 24, 24, patch_size)
#     if not lT.img_like(inp_imgs.shape):
#         return 0.0

#     def sheared_model(t):
#         sheared_img = lT.shear(inp_imgs, t, axis)
#         sheared_graph = lT.patch_tensor(sheared_img, patch_size)
#         graph.node_features = sheared_graph
#         _, z, _, _, _, _  = model(graph, training=False, tau=1.0)
#         z_restore = lT.restore_tensor(z.node_features, batch_size, 1, 24, 24, 4)
#         if lT.img_like(z_restore.shape):
#             z_restore = lT.shear(z_restore, -t, axis)
#         return z_restore

#     t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
#     lie_deriv = jvp(sheared_model, t, torch.ones_like(t))
#     return lie_deriv


# def stretch_lie_deriv(model, graph, axis="x", patch_size: int = 4):
#     """Lie derivative of model with respect to stretch, assumes scalar output"""
#     batch_size = graph.node_features.shape[0]
#     inp_imgs = lT.restore_tensor(graph.node_features, batch_size, 1, 24, 24, patch_size)
#     if not lT.img_like(inp_imgs.shape):
#         return 0.0

#     def stretched_model(t):
#         stretched_img = lT.stretch(inp_imgs, t, axis)
#         stretched_graph = lT.patch_tensor(stretched_img, patch_size)
#         graph.node_features = stretched_graph
#         _, z, _, _, _, _  = model(graph, training=False, tau=1.0)
#         z_restore = lT.restore_tensor(z.node_features, batch_size, 1, 24, 24, 4)
#         if lT.img_like(z_restore.shape):
#             z_restore = lT.stretch(z_restore, -t, axis)
#         return z_restore

#     t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
#     lie_deriv = jvp(stretched_model, t, torch.ones_like(t))
#     return lie_deriv


# def saturate_lie_deriv(model, graph, patch_size: int = 4):
#     """Lie derivative of model with respect to saturation, assumes scalar
#     output"""
#     batch_size = graph.node_features.shape[0]
#     inp_imgs = lT.restore_tensor(graph.node_features, batch_size, 1, 24, 24, patch_size)
#     if not lT.img_like(inp_imgs.shape):
#         return 0.0

#     def saturated_model(t):
#         saturated_img = lT.saturate(inp_imgs, t)
#         saturated_graph = lT.patch_tensor(saturated_img, patch_size)
#         graph.node_features = saturated_graph
#         _, z, _, _, _, _  = model(graph, training=False, tau=1.0)
#         z_restore = lT.restore_tensor(z.node_features, batch_size, 1, 24, 24, 4)
#         if lT.img_like(z_restore.shape):
#             z_restore = lT.saturate(z_restore, -t)
#         return z_restore

#     t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
#     lie_deriv = jvp(saturated_model, t, torch.ones_like(t))
#     return lie_deriv
