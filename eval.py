import argparse
import sys
import os
import glob
import numpy as np
import torch
from torch import nn
from torch import Tensor
import yaml
from model import RawNet
from torch.nn import functional as F
import librosa
import json

RESULT_FILE = "classification_results.txt"

def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

def load_sample(sample_path, max_len=96000):
    y_list = []
    y, sr = librosa.load(sample_path, sr=None)
    
    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)
        
    if len(y) <= 96000:
        return [Tensor(pad(y, max_len))]
        
    for i in range(int(len(y) / 96000)):
        if (i + 1) == range(int(len(y) / 96000)):
            y_seg = y[i * 96000:]
        else:
            y_seg = y[i * 96000: (i + 1) * 96000]

        y_pad = pad(y_seg, max_len)
        y_inp = Tensor(y_pad)
        y_list.append(y_inp)
        
    return y_list

def analyze_audio(input_path, model):
    """ 加载单个音频文件并进行推理 """
    out_list_multi = []
    out_list_binary = []

    for m_batch in load_sample(input_path):
        m_batch = m_batch.to(device=device, dtype=torch.float).unsqueeze(0)
        logits, multi_logits = model(m_batch)

        probs = F.softmax(logits, dim=-1)
        probs_multi = F.softmax(multi_logits, dim=-1)

        out_list_multi.append(probs_multi.tolist()[0])
        out_list_binary.append(probs.tolist()[0])

    result_multi = np.average(out_list_multi, axis=0).tolist()
    result_binary = np.average(out_list_binary, axis=0).tolist()

    return result_binary, result_multi

def analyze_audio_folder(folder_path, model, output_file=RESULT_FILE):
    """ 批量处理 .wav 文件，并覆盖已有的 classification_results.txt """
    # 获取文件夹内所有 .wav 文件
    audio_files = glob.glob(os.path.join(folder_path, "*.wav"))
    if not audio_files:
        print(f"No .wav files found in {folder_path}")
        return

    # **覆盖已有的 classification_results.txt，并写入表头**
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Filename, Fake Probability, Real Probability, gt, wavegrad, diffwave, parallel wave gan, wavernn, wavenet, melgan\n")

    results = []
    for audio_file in audio_files:
        file_name = os.path.basename(audio_file)
        result_binary, result_multi = analyze_audio(audio_file, model)

        # 格式化结果
        result_line = f"{file_name}, {result_binary[0]:.4f}, {result_binary[1]:.4f}, " + \
                      ", ".join(f"{r:.4f}" for r in result_multi) + "\n"
        results.append(result_line)

        print(f"Processed: {file_name} -> Fake: {result_binary[0]:.4f}, Real: {result_binary[1]:.4f}")

    # **将所有结果写入文件**
    with open(output_file, "a", encoding="utf-8") as f:
        f.writelines(results)

    print(f"Classification results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to an audio file or a folder containing .wav files')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file')
    args = parser.parse_args()

    input_path = args.input_path
    model_path = args.model_path

    # load model config
    dir_yaml = 'model_config_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    # load cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # init model
    model = RawNet(parser1['model'], device)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print('Model loaded : {}'.format(model_path))

    model.eval()

    # 判断是单个文件还是整个文件夹
    if os.path.isdir(input_path):
        analyze_audio_folder(input_path, model)
    elif os.path.isfile(input_path) and input_path.lower().endswith(".wav"):
        result_binary, result_multi = analyze_audio(input_path, model)

        print(f"Binary classification result: Fake: {result_binary[0]:.4f}, Real: {result_binary[1]:.4f}")
        print(f"Multi classification result: {result_multi}")
    else:
        print("Invalid input. Please enter a valid file or folder path.")
