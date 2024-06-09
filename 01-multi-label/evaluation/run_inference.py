import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
from torchinfo import summary
import numpy as np
import math

class InferencePerformance:
    def __init__(self, model_str, avg_mem_usage, latency, latency_std, throughput, num_params, file_size=None):
        self.model_str = model_str
        self.avg_mem_usage = avg_mem_usage
        self.latency = latency
        self.latency_std = latency_std
        self.throughput = throughput
        self.num_params = num_params
        self.file_size = file_size

def test_inference_cpu(args, model, model_str, dataloader_test, device, logger):
    model.to(device)
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)

    for _ in range(10):
        _ = model(dummy_input)
    
    latencies = []

    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(dataloader_test)):
            targets = targets.to(device)
            y_test = torch.cat((y_test, targets), 0)

            start_time = time.time()  # Start timing

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

            end_time = time.time()  # End timing
            latencies.append(end_time - start_time)  # Calculate latency for each batch

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    total_time = sum(latencies)
    total_samples = len(dataloader_test.dataset)
    throughput = total_samples / total_time

    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert bytes to GB
    peak_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # Convert bytes to GB
    torch.cuda.reset_peak_memory_stats(device)  # Reset after capturing to handle the next loop properly

    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # TODO: Fill in code for avg_mem_usage: peak memory usage and peak memory reserved
    # TODO: Fill in code for latency: mean_syn*1000, std_syn
    # TODO: Fill code for throughput

    return InferencePerformance(
        model_str=model_str,
        avg_mem_usage={'peak_memory': peak_memory, 'peak_memory_reserved': peak_memory_reserved},
        latency=mean_latency*1000, 
        latency_std=std_latency,
        throughput=throughput,
        num_params=num_params
    )


def test_inference_gpu(args, model, model_str, dataloader_test, device, logger):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    num_batches = math.ceil(len(dataloader_test.dataset) / args.batch_size)
    timings=np.zeros((num_batches,1))
    
    # gpu warm-ups
    for _ in range(10):
        _ = model(dummy_input)
    
    memory_usage = []
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(dataloader_test)):
            targets = targets.to(device)
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = samples.view(-1, c, h, w).to(device)
            starter.record()
            out = model(varInput)
            out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)
            ender.record()

            # awaiting gpu sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

            # track memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                memory_peak_allocated = torch.cuda.max_memory_allocated(device)
                memory_peak_reserved = torch.cuda.max_memory_reserved(device)
                memory_usage.append((memory_allocated, memory_reserved, memory_peak_allocated, memory_peak_reserved))
    
    # calculate throughput
    mean_syn = np.sum(timings) / len(dataloader_test.dataset)/32
    std_syn = np.std(timings)
    logger.info(f"Throughput: {mean_syn} samples/second")
    logger.info(f"Latency: {mean_syn*1000}+-{std_syn} ms")
    
    # track memory usage on gpu
    if torch.cuda.is_available():
        average_memory_allocated = sum(m[0] for m in memory_usage) / len(memory_usage) / (1024 * 1024)
        average_memory_reserved = sum(m[1] for m in memory_usage) / len(memory_usage) / (1024 * 1024)
        average_peak_memory_allocated = sum(m[2] for m in memory_usage) / len(memory_usage) / (1024 * 1024)
        average_peak_memory_reserved = sum(m[3] for m in memory_usage) / len(memory_usage) / (1024 * 1024)
        
        logger.info(f"Average Allocated Memory: {average_memory_allocated:.2f} MB")
        logger.info(f"Average Reserved Memory: {average_memory_reserved:.2f} MB")
        logger.info(f"Average Peak Allocated Memory: {average_peak_memory_allocated:.2f} MB")
        logger.info(f"Average Peak Reserved Memory: {average_peak_memory_reserved:.2f} MB")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    return InferencePerformance(
        model_str=model_str,
        avg_mem_usage={'peak_memory': average_peak_memory_allocated, 'peak_memory_reserved': average_peak_memory_reserved},
        latency=mean_syn*1000,
        latency_std=std_syn,
        throughput=mean_syn,
        num_params=num_params
    )

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