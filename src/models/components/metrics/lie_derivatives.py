import torch
import lie_transforms as lT

def jvp(f, x, u):
    """Jacobian vector product Df(x)u vs typical autograd VJP vTDF(x).
    Uses two backwards passes: computes (vTDF(x))u and then derivative wrt to v to get DF(x)u"""
    with torch.enable_grad():
        y = f(x)
        v = torch.ones_like(
            y, requires_grad=True
        )  # Dummy variable (could take any value)
        vJ = torch.autograd.grad(y, [x], [v], create_graph=True)
        Ju = torch.autograd.grad(vJ, [v], [u], create_graph=True)
        return Ju[0]


def translation_lie_deriv(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to translation vector, output can be a scalar or an image"""
    # vector = vector.to(inp_imgs.device)
    if not lT.img_like(inp_imgs.shape):
        return 0.0

    def shifted_model(t):
        # print("Input shape",inp_imgs.shape)
        shifted_img = lT.translate(inp_imgs, t, axis)
        z = model(shifted_img)
        # print("Output shape",z.shape)
        # if model produces an output image, shift it back
        if lT.img_like(z.shape):
            z = lT.translate(z, -t, axis)
        # print('zshape',z.shape)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(shifted_model, t, torch.ones_like(t, requires_grad=True))
    
    return lie_deriv


def rotation_lie_deriv(model, inp_imgs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not lT.img_like(inp_imgs.shape):
        return 0.0

    def rotated_model(t):
        rotated_img = lT.rotate(inp_imgs, t)
        z = model(rotated_img)
        if lT.img_like(z.shape):
            z = lT.rotate(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
    return lie_deriv


def hyperbolic_rotation_lie_deriv(model, inp_imgs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not lT.img_like(inp_imgs.shape):
        return 0.0

    def rotated_model(t):
        rotated_img = lT.hyperbolic_rotate(inp_imgs, t)
        z = model(rotated_img)
        if lT.img_like(z.shape):
            z = lT.hyperbolic_rotate(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
    return lie_deriv


def scale_lie_deriv(model, inp_imgs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not lT.img_like(inp_imgs.shape):
        return 0.0

    def scaled_model(t):
        scaled_img = lT.scale(inp_imgs, t)
        z = model(scaled_img)
        if lT.img_like(z.shape):
            z = lT.scale(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(scaled_model, t, torch.ones_like(t))
    return lie_deriv


def shear_lie_deriv(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to shear, assumes scalar output"""
    if not lT.img_like(inp_imgs.shape):
        return 0.0

    def sheared_model(t):
        sheared_img = lT.shear(inp_imgs, t, axis)
        z = model(sheared_img)
        if lT.img_like(z.shape):
            z = lT.shear(z, -t, axis)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(sheared_model, t, torch.ones_like(t))
    return lie_deriv


def stretch_lie_deriv(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to stretch, assumes scalar output"""
    if not lT.img_like(inp_imgs.shape):
        return 0.0

    def stretched_model(t):
        stretched_img = lT.stretch(inp_imgs, t, axis)
        z = model(stretched_img)
        if lT.img_like(z.shape):
            z = lT.stretch(z, -t, axis)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(stretched_model, t, torch.ones_like(t))
    return lie_deriv


def saturate_lie_deriv(model, inp_imgs):
    """Lie derivative of model with respect to saturation, assumes scalar output"""
    if not lT.img_like(inp_imgs.shape):
        return 0.0

    def saturated_model(t):
        saturated_img = lT.saturate(inp_imgs, t)
        z = model(saturated_img)
        if lT.img_like(z.shape):
            z = lT.saturate(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(saturated_model, t, torch.ones_like(t))
    return lie_deriv