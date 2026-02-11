import torch
import matplotlib.pyplot as plt
import math

def test_query_distribution(n_queries=30):
    # 1. Lógica idêntica ao seu SuperiorDETR
    cols = math.ceil(n_queries ** 0.5)
    rows = math.ceil(n_queries / cols)

    # Gerando os eixos de 0.1 a 0.9 (evitando as bordas extremas)
    xs = torch.linspace(0.1, 0.9, steps=cols)
    ys = torch.linspace(0.1, 0.9, steps=rows)
    
    # Criando o Grid
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).view(-1, 2)
    
    # O "pulo do gato": fatiar para garantir que temos exatamente n_queries
    # mesmo que o grid cols x rows tenha gerado mais pontos
    grid = grid[:n_queries]

    # 2. Visualização
    plt.figure(figsize=(8, 8))
    
    # Desenha a área da imagem (0 a 1)
    plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, color='black', linestyle='--', alpha=0.3))
    
    # Plota as queries
    plt.scatter(grid[:, 0], grid[:, 1], c='blue', marker='o', s=100, edgecolors='black', label='Anchor Points')
    
    # Adiciona o índice de cada query para conferir a ordem de busca
    for i, (x, y) in enumerate(grid):
        plt.text(x + 0.02, y + 0.02, str(i), fontsize=9, alpha=0.7)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title(f"Distribuição Genérica: {n_queries} Queries\nGrid Calculado: {cols} colunas x {rows} linhas")
    plt.xlabel("Largura Normalizada (x)")
    plt.ylabel("Altura Normalizada (y)")
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()

# TESTE AQUI: Mude o número e execute
test_query_distribution(n_queries=30)
test_query_distribution(n_queries=50)
test_query_distribution(n_queries=15)