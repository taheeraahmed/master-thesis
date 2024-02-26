from pynvml import *
import sys

def check_gpu(logger):
    try:
        nvmlInit()
        logger.info(f"Driver Version: {nvmlSystemGetDriverVersion()}")
        deviceCount = nvmlDeviceGetCount()
        devices = []
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            logger.info(f"Device {i}: {nvmlDeviceGetName(handle)}")
            # gpu_device = GPUDevice(handle=handle, gpu_index=i)
            devices.append(nvmlDeviceGetTotalEnergyConsumption(handle))
    except:
        logger.error('NVML drivrer version not accessible')