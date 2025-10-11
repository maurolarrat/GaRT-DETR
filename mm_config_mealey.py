# config.py

# ===========================================================
# CONFIGURAÇÃO DO MULTIMODAL TRANSFORMER
# ===========================================================

CONFIG = {
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 1e-6,
    "transformer_hidden_dim": 256,
    "modal_feature_dim": 512, # Dimensão de saída do PointNet/Encoder de PCL
    "pcl_feature_dim": 9, 
    "image_size": 224,
    "sequence_length": 10,
    "num_workers": 4,
    "data_root": r"C:\Users\Micro\Documents\sourcecode\MMNTT\Mavic3",
    "dropout_rate": 0.2,
    "mealy_temp_scale": 0.5,
    "temporal_loss_weight": 0.5
}

