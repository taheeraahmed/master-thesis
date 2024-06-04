import time
import torch
import tracemalloc
import logging
from tqdm import tqdm
from torchinfo import summary

def test_inference(model, data_loader_test, device, logger):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)
    
    # start measuring peak memory usage
    tracemalloc.start()
    
    # start time for throughput measurement
    start_time = time.time()
    
    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
            targets = targets.to(device)
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = samples.view(-1, c, h, w).to(device)

            out = model(varInput)
            out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)
    
    end_time = time.time()
    
    # measure peak memory usage
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # throughput calculation
    total_time = end_time - start_time
    num_samples = len(data_loader_test.dataset)
    throughput = num_samples / total_time
    
    logger.info(f"Time taken for inference: {total_time} seconds")
    logger.info(f"Throughput: {throughput} samples/second")
    logger.info(f"Peak memory usage: {peak_memory / 1024 / 1024} MB")
    
    # model size
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)  # Convert bytes to MB
    
    logger.info(f"Number of parameters: {num_params}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    return y_test, p_test


def predict(model, batch, labels, threshold=0.5):
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Turn off gradients to speed up this part
        logits = model(batch)  # Forward pass
        print(logits.shape)
        probabilities = torch.sigmoid(logits)
        probabilities_numpy = (probabilities >= threshold).int().numpy()[0]

    print("Predictions:")
    for i, label in enumerate(probabilities_numpy):
        if label:
            print(f"\t{labels[i]}")