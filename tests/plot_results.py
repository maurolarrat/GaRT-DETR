import re
import matplotlib.pyplot as plt
import numpy as np

def parse_logs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    epoch_blocks = re.split(r"={30,} ÉPOCA (\d+)/300 ={30,}", text)[1:]
    data = {k: [] for k in ['epoch', 'loss', 'iou_global', 'msa_global', 
                            'iou_ir', 'iou_vis', 'msa_ir', 'msa_vis', 
                            'gate_ir', 'gate_vis']}

    for i in range(0, len(epoch_blocks), 2):
        content = epoch_blocks[i+1]
        if "[RESULTADOS DA ÉPOCA" in content:
            data['epoch'].append(int(epoch_blocks[i]))
            
            def find_val(metric):
                match = re.search(rf"{metric}\s+\|\s+[\d. (±)]+\s+\|\s+([\d.]+)", content)
                return float(match.group(1)) if match else 0.0

            for key in ['loss', 'iou_global', 'msa_global', 'iou_ir_avg', 'iou_vis_avg', 
                        'msa_ir_avg', 'msa_vis_avg', 'gate_ir_avg', 'gate_vis_avg']:
                clean_key = key.replace('_avg', '')
                data[clean_key].append(find_val(key.upper()))
    return data

def plot_refined_analysis(d):
    # pdf como formato padrão de fonte
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'axes.grid': True})
    epochs = np.array(d['epoch'])
    loss = np.array(d['loss'])

    # FIGURE 1: GLOBAL CONVERGENCE
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.axvspan(0, 10, color='yellow', alpha=0.15, label='Warm-up (Exist Weight 0.2)', zorder=0)
    
    ax1.plot(epochs, loss, 'r-o', markersize=3, label='Total Loss', alpha=0.8)
    ax1.set_ylabel('Loss Magnitude', color='r')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, d['iou_global'], 'k-s', linewidth=1.5, label='Global IoU')
    ax2.plot(epochs, d['msa_global'], 'g--', linewidth=1.5, label='Global MSA')
    ax2.set_ylabel('Performance Score')
    
    plt.title('Figure 1: Global Error Convergence vs. Accuracy')
    leg1 = fig1.legend(loc='upper left', bbox_to_anchor=(0.25, 0.92), framealpha=0.9)
    leg1.set_zorder(100)
    plt.tight_layout()
    
    # SALVANDO EM PDF VETORIAL (Aceita transparência e é reconhecido pelo Overleaf)
    plt.savefig('global_metrics.pdf', format='pdf', bbox_inches='tight')

    # FIGURE 2: MODAL ANALYSIS
    fig2, (ax_vis, ax_ir, ax_gate) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    def setup_subplot(ax):
        ax.axvspan(0, 10, color='yellow', alpha=0.2, label='Warm-up Period', zorder=10)
        
        ax_loss = ax.twinx()
        ax_loss.plot(epochs, loss, color='black', linestyle='-', linewidth=1.3, 
                      alpha=0.7, label='Global Loss (Trend)', zorder=15)
        ax_loss.set_ylabel('Global Loss', color='black', fontsize=9)
        
        lines, labs = ax.get_legend_handles_labels()
        lines2, labs2 = ax_loss.get_legend_handles_labels()
        
        leg = ax.legend(lines + lines2, labs + labs2, loc='upper left', 
                  bbox_to_anchor=(0.25, 0.98), framealpha=0.9, fontsize=9)
        leg.set_zorder(100) 
        
        return ax_loss

    # 1. VISIBLE BRANCH
    ax_vis.plot(epochs, d['iou_vis'], color='blue', marker='^', label='IoU Visible', markersize=4)
    ax_vis.plot(epochs, d['msa_vis'], color='blue', linestyle='--', label='MSA Visible', alpha=0.6)
    ax_vis.set_ylabel('Visible Scores', color='blue')
    ax_vis.set_title('Figure 2: Per-Modal Accuracy and Fusion Dynamics')
    setup_subplot(ax_vis)

    # 2. INFRARED BRANCH
    ax_ir.plot(epochs, d['iou_ir'], color='red', marker='v', label='IoU Infrared', markersize=4)
    ax_ir.plot(epochs, d['msa_ir'], color='red', linestyle='--', label='MSA Infrared', alpha=0.6)
    ax_ir.set_ylabel('Infrared Scores', color='red')
    setup_subplot(ax_ir)

    # 3. GATING DYNAMICS
    gate_v = np.array(d['gate_vis'])
    gate_i = np.array(d['gate_ir'])
    total_gate = gate_v + gate_i + 1e-8
    
    ax_gate.stackplot(epochs, gate_v/total_gate, gate_i/total_gate, 
                      labels=['Vis Contribution', 'IR Contribution'],
                      colors=['#4169E1', '#CD5C5C'], alpha=0.8, zorder=1)
    
    ax_gate.set_ylabel('Fusion Weight Ratio')
    ax_gate.set_ylim(0, 1)
    ax_gate.set_xlabel('Epoch')
    
    ax_gate.axhline(0.5, color='yellow', linestyle='--', linewidth=1.5, zorder=5)
    
    setup_subplot(ax_gate)

    plt.tight_layout()
    # SALVANDO EM PDF VETORIAL
    plt.savefig('modal_dynamics_final.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# Execução
d = parse_logs('log_train_and_validation.txt')
plot_refined_analysis(d)
