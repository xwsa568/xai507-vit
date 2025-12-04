# main.py
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import cfg
from dataloader import get_dataloaders
from model import VisionTransformer
from train import train_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_results(histories):
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        plt.plot(hist['train_loss'], label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation Accuracy Plot
    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist['val_acc'], label=name)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_result.png')
    print("Graph saved as comparison_result.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'baseline', 'rope', 'custom'])
    args = parser.parse_args()

    set_seed(cfg.seed)
    
    # Loader가 3개 리턴됨
    train_loader, val_loader, test_loader = get_dataloaders()
    
    modes = ['baseline', 'rope', 'custom'] if args.mode == 'all' else [args.mode]
    histories = {}
    test_results = {}

    for mode in modes:
        print(f"\n=== Training with {mode.upper()} PE ===")
        set_seed(cfg.seed) 
        
        model = VisionTransformer(pe_type=mode)
        # train_model 이제 history와 test_acc 두 개를 반환함
        hist, final_test_acc = train_model(model, train_loader, val_loader, test_loader, mode)
        
        histories[mode] = hist
        test_results[mode] = final_test_acc

    # 최종 결과 요약 출력
    print("\n" + "="*30)
    print("      FINAL TEST RESULTS      ")
    print("="*30)
    for mode, acc in test_results.items():
        print(f"{mode.upper():<10}: {acc:.2f}%")
    print("="*30)

    if args.mode == 'all':
        plot_results(histories)