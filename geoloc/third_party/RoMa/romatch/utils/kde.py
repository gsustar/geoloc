import torch


def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

# def kde(x, std=0.1, half=True, down=None):
#     if half:
#         x = x.half()
    
#     def cdist_onnx(a, b):
#         a2 = (a ** 2).sum(dim=-1, keepdim=True)
#         b2 = (b ** 2).sum(dim=-1, keepdim=True).T
#         ab = a @ b.T
#         return torch.sqrt((a2 + b2 - 2 * ab).clamp(min=0))
    
#     if down is not None:
#         scores = (-cdist_onnx(x, x[::down]) ** 2 / (2 * std ** 2)).exp()
#     else:
#         scores = (-cdist_onnx(x, x) ** 2 / (2 * std ** 2)).exp()
    
#     density = scores.sum(dim=-1)
#     return density