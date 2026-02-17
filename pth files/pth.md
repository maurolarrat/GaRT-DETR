Arquivos gerados após o treino e validação do modelo.

Antes de usar estes arquivos, certifique-se de renomeá-los para o nome usado nos scripts.

### Model with the best global and individual multimodal results, and with inferior results when faced with the absence of one of the modalities
Link para download: https://drive.google.com/file/d/1laDVIEULgIa0FXeGq3DGFwmXb1SumitP/view?usp=drive_link

### Table I: Final Performance Evaluation on Anti-UAV300 Test Set

| Metric | Test Dataset (Unseen) |
| :--- | :--- |
| LOSS | **4.1325** |
| IOU_GLOBAL | **0.6047** |
| MSA_GLOBAL | **0.7998** |
| GATE_VIS_AVG | **0.5576 (±0.0000)** |
| GATE_IR_AVG | **0.0643 (±0.0241)** |
| IOU_VIS_AVG | **0.6085** |
| IOU_IR_AVG | **0.5706** |
| MSA_VIS_AVG | **0.8037** |
| MSA_IR_AVG | **0.7646** |

---

### Table II: Comparative mSA (%) performance against State-of-the-Art trackers on the Anti-UAV300 Test Set.

| Tracker | mSA (IR) | mSA (VIS) |
| :--- | :---: | :---: |
| SiamRPN++LT | 65.84 | 67.15 |
| GlobalTrack | 72.00 | 67.28 |
| SiamRCNN | 74.33 | 74.32 |
| **SuperiorDETR (Ours)** | **76.46** | **80.37** |

---

### Table III: Performance Under Single-Modality Constraints (Ablation Study) on the Anti-UAV300 Test Set.

| Metric | Visible Only | IR Only |
| :--- | :---: | :---: |
| IOU_GLOBAL | 0.4993 | 0.2499 |
| MSA_GLOBAL | 0.5831 | 0.1194 |
| IOU_VIS (Active) | **0.5047** | 0.2326* |
| IOU_IR (Active) | 0.0151* | **0.4016** |
| MSA_VIS (Active) | **0.5896** | 0.0938* |
| MSA_IR (Active) | 0.0031* | **0.3407** |
| GATE_IR_AVG | **0.0062** | 0.0611 |

*\*Indicates performance of the deactivated modality.*
