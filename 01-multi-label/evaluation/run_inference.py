import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchinfo import summary
import numpy as np
import math
import torch
import numpy as np
from memory_profiler import memory_usage
import math
from tqdm import tqdm
import timeit

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
    raise NotImplementedError("CPU inference is not implemented yet")


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
