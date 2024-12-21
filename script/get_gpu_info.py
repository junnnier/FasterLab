import pynvml


def get_gpu_mem_info(gpu_id=0):
    """
    Args:
        gpu_id: 显卡 ID
    Returns: gpuname gpu名称，total 所有的显存，used 当前使用的显存, free 可使用的显存, use_ratio利用率，power功耗
    """
    pynvml.nvmlInit()
    if 0 <= gpu_id < pynvml.nvmlDeviceGetCount():
        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        gpuname = pynvml.nvmlDeviceGetName(handler)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handler)
        powerusage = pynvml.nvmlDeviceGetPowerUsage(handler)
        total = round(meminfo.total / 1024 / 1024, 2)
        used = round(meminfo.used / 1024 / 1024, 2)
        free = round(meminfo.free / 1024 / 1024, 2)
        use_ratio = utilization.gpu
        power = round(powerusage / 1000, 1)
    else:
        print(r'gpu_id {} 显卡不存在!'.format(gpu_id))
        return 0, 0, 0, 0, 0, 0
    return gpuname, total, used, free, use_ratio, power


if __name__ == '__main__':
    gpu_name, gpu_mem_total, gpu_mem_used, gpu_mem_free, gpu_ratio, gpu_power = get_gpu_mem_info(gpu_id=0)
    print("{}\tTotal: {} MB\tUsed: {} MB\tFree: {} MB\tRatio: {}%\tPower: {}w".format(gpu_name, gpu_mem_total, gpu_mem_used, gpu_mem_free, gpu_ratio, gpu_power))