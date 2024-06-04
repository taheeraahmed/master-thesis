import torch
import tqdm
import time

def test_inference(model, data_loader_test, device):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    model.eval()

    y_test = torch.FloatTensor().cuda()
    p_test = torch.FloatTensor().cuda()
    start_time = time.time()
    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
            targets = targets.cuda()
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = torch.autograd.Variable(
                samples.view(-1, c, h, w).cuda())

            out = model(varInput)
            out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)
    end_time = time.time()
    print(f"Time taken for inference: {end_time - start_time} seconds")
    return y_test, p_test


def predict(model, batch, threshold=0.2):
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Turn off gradients to speed up this part
        logits = model(batch)  # Forward pass
        print(logits.shape)
        probabilities = torch.sigmoid(logits)
        labels = (probabilities >= threshold).int().numpy()[0]

    pred_idx = [idx for idx, _ in enumerate(labels) if labels[idx] == 1]
    print(pred_idx)
    for i in range(len(pred_idx)):
        print(labels[i])
