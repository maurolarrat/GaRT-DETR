Esta é a documentação atualizada do módulo `dataloader.py`, refletindo as novas lógicas de amostragem e os limites de frames por sequência para o dataset **Anti-UAV 300**.

---

# Documentação do Dataloader: AntiUAVRGBTDataset

Este módulo implementa o carregamento de dados multimodal (RGB + Infravermelho) otimizado para o treinamento de modelos de rastreamento temporal. A nova versão introduz maior controle sobre a densidade da amostragem e limites de memória.

## 1. Modificações e Melhorias Recentes

Diferente da versão anterior, este dataloader agora inclui:

* **`max_frames_per_seq`**: Um limitador que restringe quantos frames de uma sequência original serão considerados, útil para equilibrar o dataset ou acelerar épocas de treino.
* **Amostragem Uniforme (Linspace)**: Quando uma sequência é maior que a janela temporal desejada, o código seleciona frames distribuídos uniformemente, garantindo que o modelo veja o "movimento" completo do drone em vez de apenas um trecho curto.

---

## 2. Parâmetros de Inicialização

| Parâmetro | Tipo | Descrição |
| --- | --- | --- |
| `root_dir` | `str` | Caminho raiz do dataset Anti-UAV. |
| `split` | `str` | Subconjunto de dados (ex: "train", "val", "test"). |
| `temporal_window` | `int` | Tamanho fixo da sequência de saída (ex: 30 frames). |
| `max_frames_per_seq` | `int` | Limite máximo de frames lidos por pasta de sequência. |

---

## 3. Lógica de Seleção de Frames

O método `__getitem__` agora opera sob uma hierarquia de decisão para garantir que o tensor de saída tenha sempre o mesmo tamanho (`temporal_window`):

1. **Filtragem**: Apenas frames com Ground Truth (GT) válido em ambas as modalidades são carregados.
2. **Truncamento**: Se definido, a sequência é cortada em `max_frames_per_seq`.
3. **Caso A (Sequência Curta)**: Se a sequência disponível for menor que a janela, o sistema preenche o restante repetindo o último frame válido (*padding*).
4. **Caso B (Sequência Longa)**:
* Um segmento aleatório é escolhido dentro da sequência.
* O código utiliza `np.linspace` para extrair `temporal_window` frames desse segmento de forma equidistante.



---

## 4. Estrutura do Tensor de Entrada (`x_input`)

O dataloader prepara os dados para uma arquitetura de fusão precoce (*early fusion*) ou processamento paralelo:

* **Composição**: As imagens RGB (3 canais) e IR (1 canal) são empilhadas no eixo da dimensão de canais.
* **Shape Final**: `[Janela_Temporal, 4, H, W]`
* Canal 0, 1, 2: Informação visual (Red, Green, Blue).
* Canal 3: Informação térmica (Infrared).



---

## 5. Normalização de Coordenadas (Ground Truth)

As Bounding Boxes são processadas para o formato de centro relativo para facilitar o cálculo de perdas de regressão:

Isso garante que, independentemente da resolução da câmera (que pode variar entre o sensor visível e o térmico), os valores estejam sempre entre .

---

## 6. Funções Auxiliares: `collate_fn_superior`

Esta função é passada para o `DataLoader` do PyTorch para organizar o batch. Ela transforma uma lista de dicionários (retornada pelo `__getitem__`) em um único dicionário de tensores batched:

* **Inputs e Boxes**: Agrupados em tensores de dimensão `[Batch, Window, ...]`.
* **Metadados**: Os nomes das sequências (`seq_names`) são mantidos como uma lista de strings para fins de depuração e avaliação.

---
