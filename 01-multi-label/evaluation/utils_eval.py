import os

MODEL_STR_TO_TABLE = {
    "vit_in1k": "ViT-IN1K",
    "swin_in22k": "Swin-IN22K",
    "densenet121": "DenseNet121-IN1K",
    "swin_simim": "Swin-SimMIM"
}

def get_file_size(file_path):
    file_size = os.path.getsize(file_path)
    return file_size

def three_decimal_places(num):
    return round(num,3)

def generate_latex_table(inference_performances):
    # Start of the LaTeX table
    latex_str = r'''\begin{table}[h!]
    \centering
    \caption[Inference time for each model.]{Inference time (IT) for the models, given the ViT, swin-transformer and DenseNet121 pre-trained on ImageNet (IN) and the swin-transformer pre-trained using SimMIM. There are two inference times, one per batch and one average given there are 25,596 samples.}
    \label{tab:5-inference-time}
    \begin{tabular}{lcccc}
    \toprule
    Metric'''

    # Adding model names as column headers
    for ip in inference_performances:
        latex_str += f' & {MODEL_STR_TO_TABLE[ip.model_str]}'
    latex_str += r'''\\
    \midrule'''

    # Adding model size row
    latex_str += r'Model size (MB)'
    for ip in inference_performances:
        model_size = ip.file_size if ip.file_size is not None else 'N/A'
        latex_str += f' & {model_size}'
    latex_str += r' \\'

    # Adding number of parameters row
    latex_str += r'Params'
    for ip in inference_performances:
        latex_str += f' & {ip.num_params}'
    latex_str += r' \\ '

    # Adding throughput row
    latex_str += r'Latecy (ms)'
    for ip in inference_performances:
        latex_str += f' & {three_decimal_places(ip.latency):.2f} $\pm$ {three_decimal_places(ip.latency_std):.2f}'
    latex_str += r' \\ \hline'

    # Adding average memory usage row
    latex_str += r'Peak memory allocated (Allocated, Reserved)'
    for ip in inference_performances:
        avg_mem_usage = f"({three_decimal_places(ip.avg_mem_usage['peak_memory'])}, {three_decimal_places(ip.avg_mem_usage['peak_memory_reserved'])}"
        latex_str += f' & {avg_mem_usage}'
    latex_str += r' \\ \hline'

    latex_str += r'''
    \bottomrule
    \end{tabular}
\end{table}'''

    return latex_str