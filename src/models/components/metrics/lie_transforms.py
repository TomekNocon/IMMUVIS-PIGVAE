import torch
import torch.nn.functional as F
import numpy as np


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        ix_nw = IW - 1 - (IW - 1 - ix_nw.abs()).abs()
        iy_nw = IH - 1 - (IH - 1 - iy_nw.abs()).abs()

        ix_ne = IW - 1 - (IW - 1 - ix_ne.abs()).abs()
        iy_ne = IH - 1 - (IH - 1 - iy_ne.abs()).abs()

        ix_sw = IW - 1 - (IW - 1 - ix_sw.abs()).abs()
        iy_sw = IH - 1 - (IH - 1 - iy_sw.abs()).abs()

        ix_se = IW - 1 - (IW - 1 - ix_se.abs()).abs()
        iy_se = IH - 1 - (IH - 1 - iy_se.abs()).abs()

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)
    )

    out_val = (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W)
        + ne_val.view(N, C, H, W) * ne.view(N, 1, H, W)
        + sw_val.view(N, C, H, W) * sw.view(N, 1, H, W)
        + se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )

    return out_val


def img_like(img_shape):
    bchw = len(img_shape) == 4 and img_shape[-2:] != (1, 1)
    is_square = int(int(np.sqrt(img_shape[1])) + 0.5) ** 2 == img_shape[1]
    is_one_off_square = int(int(np.sqrt(img_shape[1])) + 0.5) ** 2 == img_shape[1] - 1
    is_two_off_square = int(int(np.sqrt(img_shape[1])) + 0.5) ** 2 == img_shape[1] - 2
    bnc = (
        len(img_shape) == 3
        and img_shape[1] != 1
        and (is_square or is_one_off_square or is_two_off_square)
    )
    return bchw or bnc


def num_tokens(img_shape):
    if len(img_shape) == 4 and img_shape[-2:] != (1, 1):
        return 0
    is_one_off_square = int(int(np.sqrt(img_shape[1])) + 0.5) ** 2 == img_shape[1] - 1
    is_two_off_square = int(int(np.sqrt(img_shape[1])) + 0.5) ** 2 == img_shape[1] - 2
    return int(is_one_off_square * 1 or is_two_off_square * 2)


def bnc2bchw(bnc, num_tokens):
    b, n, c = bnc.shape
    h = w = int(np.sqrt(n))
    extra = bnc[:, :num_tokens, :]
    img = bnc[:, num_tokens:, :]
    return img.reshape(b, h, w, c).permute(0, 3, 1, 2), extra


def bchw2bnc(bchw, tokens):
    b, c, h, w = bchw.shape
    n = h * w
    bnc = bchw.permute(0, 2, 3, 1).reshape(b, n, c)
    return torch.cat([tokens, bnc], dim=1)  # assumes tokens are at the start


def affine_transform(affineMatrices, img):
    assert img_like(img.shape)
    if len(img.shape) == 3:
        ntokens = num_tokens(img.shape)
        x, extra = bnc2bchw(img, ntokens)
    else:
        x = img
    flowgrid = F.affine_grid(
        affineMatrices, size=x.size(), align_corners=True
    )  # .double()
    # uses manual grid sample implementation to be able to compute 2nd derivatives
    # img_out = F.grid_sample(img, flowgrid,padding_mode="reflection",align_corners=True)
    transformed = grid_sample(x, flowgrid)
    if len(img.shape) == 3:
        transformed = bchw2bnc(transformed, extra)
    return transformed


def translate(img, t, axis="x"):
    """Translates an image by a fraction of the size (sx,sy) in (0,1)"""
    affineMatrices = torch.zeros(img.shape[0], 2, 3).to(img.device)
    affineMatrices[:, 0, 0] = 1
    affineMatrices[:, 1, 1] = 1
    if axis == "x":
        affineMatrices[:, 0, 2] = t
    else:
        affineMatrices[:, 1, 2] = t
    return affine_transform(affineMatrices, img)


def rotate(img, angle):
    """Rotates an image by angle"""
    affineMatrices = torch.zeros(img.shape[0], 2, 3).to(img.device)
    affineMatrices[:, 0, 0] = torch.cos(angle)
    affineMatrices[:, 0, 1] = torch.sin(angle)
    affineMatrices[:, 1, 0] = -torch.sin(angle)
    affineMatrices[:, 1, 1] = torch.cos(angle)
    return affine_transform(affineMatrices, img)


def shear(img, t, axis="x"):
    """Shear an image by an amount t"""
    affineMatrices = torch.zeros(img.shape[0], 2, 3).to(img.device)
    affineMatrices[:, 0, 0] = 1
    affineMatrices[:, 1, 1] = 1
    if axis == "x":
        affineMatrices[:, 0, 1] = t
        affineMatrices[:, 1, 0] = 0
    else:
        affineMatrices[:, 0, 1] = 0
        affineMatrices[:, 1, 0] = t
    return affine_transform(affineMatrices, img)


def stretch(img, x, axis="x"):
    """
    Stretches the image along the specified axis by (1 + x),
    where x is a fractional stretch factor.
    """
    affineMatrices = torch.zeros(img.shape[0], 2, 3).to(img.device)
    x = x.to(img.device)
    # Set both diagonal entries to preserve the image unless stretched
    affineMatrices[:, 0, 0] = 1.0
    affineMatrices[:, 1, 1] = 1.0

    if axis == "x":
        affineMatrices[:, 0, 0] *= 1 + x  # stretch x-axis
    else:
        affineMatrices[:, 1, 1] *= 1 + x  # stretch y-axis

    return affine_transform(affineMatrices, img)


def hyperbolic_rotate(img, angle):
    affineMatrices = torch.zeros(img.shape[0], 2, 3).to(img.device)
    affineMatrices[:, 0, 0] = torch.cosh(angle)
    affineMatrices[:, 0, 1] = torch.sinh(angle)
    affineMatrices[:, 1, 0] = torch.sinh(angle)
    affineMatrices[:, 1, 1] = torch.cosh(angle)
    return affine_transform(affineMatrices, img)


def scale(img, s):
    affineMatrices = torch.zeros(img.shape[0], 2, 3).to(img.device)
    affineMatrices[:, 0, 0] = 1 - s
    affineMatrices[:, 1, 1] = 1 - s
    return affine_transform(affineMatrices, img)


def saturate(img, t):
    img = img.clone()
    t = t.to(img.device)
    img *= 1 + t
    return img


def patch_tensor(x: torch.Tensor, patch_size: int) -> np.array:
    # x -> B c h w
    # bs, c, h, w = x.shape
    bs, c, h, w = x.shape
    x = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)(x)
    # x -> B (c*p*p) L
    # Reshaping into the shape we want
    a = x.view(bs, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
    a = a.view(bs, -1, c * patch_size * patch_size)
    # a -> ( B no.of patches c p p )
    return a


def restore_tensor(
    a: np.array,
    bs: int,
    c: int,
    h: int,
    w: int,
    patch_size: int,
    to_tensor: bool = False,
) -> np.array:
    # Step 1: Reshape a to match the patch grid layout
    a = a.view(bs, -1, c, patch_size, patch_size)
    # a -> (B, num_patches, C, patch_size, patch_size)

    # Step 2: Reshape back to (B, C, H, W) by folding the patches
    # Calculate the grid size (L) which is the number of patches in each row and column
    grid_size = int(
        a.size(1) ** 0.5
    )  # Assumes square grid (height = width for the patches)

    # Unfold back into the original image size
    a = a.view(bs, grid_size, grid_size, c, patch_size, patch_size)
    # a -> (B, grid_size, grid_size, C, patch_size, patch_size)

    # Step 3: Permute and reshape to get back to the image format (B, C, H, W)
    x_reconstructed = a.permute(0, 3, 1, 4, 2, 5).contiguous()
    # x_reconstructed -> (B, C, grid_size, patch_size, grid_size, patch_size)

    x_reconstructed = x_reconstructed.view(bs, c, h, w)
    # x_reconstructed -> (B, C, H, W)
    # if to_tensor:
    #     x_reconstructed = x_reconstructed.clone()

    return x_reconstructed
