import torch

def run_inference(model, model_str, input_tensor, device, tta=True):
    # add extra batch dimension
    samples = input_tensor.unsqueeze(0)

    if tta:
        with torch.no_grad():
            bs, n_crops, c, h, w = samples.size()
            # flatten out the batch and n_crops dimensions
            input = samples.view(-1, c, h, w).to(device)
            out = model(input)
            out = torch.sigmoid(out)
            # average the outputs of the TTA crops
            out_mean = out.view(bs, n_crops, -1).mean(1)
    
    return out_mean
