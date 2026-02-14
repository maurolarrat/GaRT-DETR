# Guia de Treinamento e Critério Multimodal: SuperiorDETR

Este documento detalha o pipeline de treinamento do modelo **SuperiorDETR**, explicando como as predições de dois sensores distintos (Visível e Infravermelho) são pareadas, calculadas e otimizadas através de uma função de perda (*loss*) composta.

## 1. Configurações de Entrada e Normalização

O treinamento utiliza transformações específicas para cada braço do modelo, garantindo que os dados cheguem no formato esperado pelos backbones pré-treinados:

* **RGB:** Normalização padrão ImageNet para a ResNet18.
* **Infravermelho (IR):** Normalização monocanal para a EfficientNet adaptada.
* **Coordenadas:** O modelo trabalha internamente com coordenadas normalizadas  no formato `cxcywh` (centro x, centro y, largura, altura), mas o critério as converte para `xyxy` para cálculos geométricos.

---

## 2. O Coração do Treino: `MultimodalCriterion`

O `MultimodalCriterion` não é apenas uma função de perda; ele atua como um **Matcher Dinâmico** que decide qual "pergunta" (query) do Transformer melhor representa o drone em cada frame.

### A. Utilitário Geométrico: Generalized IoU (GIoU)

Diferente do IoU padrão, que é zero se as caixas não se sobrepuserem, o **GIoU** fornece um gradiente mesmo quando as caixas estão distantes.

* **Cálculo:** Ele encontra o menor retângulo que envolve ambas as caixas (convexhull) e penaliza a área vazia entre elas.
* **Importância:** Vital para o início do treino, quando as predições ainda estão "espalhadas" pela imagem.

### B. O Matcher Multimodal

Como temos 20 queries, precisamos saber qual delas deve ser comparada com o Ground Truth (GT).

1. **Média de Predição:** O critério calcula a média entre a predição Visível e IR (`p_mean_xyxy`) para estabilizar o pareamento.
2. **Cálculo de Distância:** Calcula a distância L1 entre essa média e o objeto real.
3. **Seleção:** A função `dist.argmin(dim=-1)` seleciona a query de menor erro. Apenas essa query será "punida" por errar a caixa; as outras serão "punidas" apenas se afirmarem que o drone está lá (classificação).

### C. Decomposição da Loss (Função de Perda)

A perda total é uma soma ponderada de vários fatores:

| Componente | Peso | Função | Objetivo |
| --- | --- | --- | --- |
| **L1 Loss** | 5.0 | `F.l1_loss` | Precisão absoluta dos centros e dimensões. |
| **GIoU Loss** | 2.0 | `1 - GIoU` | Alinhamento geométrico e sobreposição de área. |
| **BCE Vis/IR** | 0.33 | `BCEWithLogits` | Confiança individual de cada sensor. |
| **BCE Global** | 0.33 | `BCEWithLogits` | Confiança de que o drone existe na cena (fusão). |
| **Background** | 0.1 | `pred^2` | Silenciar queries que não encontraram nada. |

---

## 3. Estratégias de Treinamento Avançadas

### Warm-up de Existência (Épocas 0-10)

Nas primeiras 10 épocas, o modelo é instável. Para evitar que ele aprenda a simplesmente "desistir" (zerar a confiança) antes de aprender a localizar, aplicamos um peso reduzido (**0.2**) na classificação.

* **Foco inicial:** Aprender a colocar a caixa em cima do drone.
* **Foco posterior:** Aprender a dizer se o drone está visível ou ocluso.

### Pesagem de Métricas via Gates

Na fase de validação, as métricas globais (`iou_global`, `msa_global`) não são médias aritméticas simples. Elas são **ponderadas pelos scores de Gate** vindo do Backbone:



Isso reflete a performance real do modelo: se ele deu mais importância ao IR em um frame noturno, o erro do IR impactará mais a métrica global.

---

## 4. Loop de Execução e Estabilidade

### Gradient Clipping

O script utiliza `torch.nn.utils.clip_grad_norm_` com `max_norm=0.1`. Isso é crucial em modelos baseados em Transformer (DETR) para evitar que gradientes explosivos destruam os pesos aprendidos durante o cálculo da atenção cruzada.

### Persistência e Checkpointing

* **Checkpoint Regular:** Salvo a cada época para permitir retomada (`RESUME_PATH`).
* **Best Model:** Salva o estado do modelo sempre que o **MSA Global** (Mean Success Area) atinge um novo recorde.

---

## 5. Como ler o Log do TQDM

Durante o treino, o postfix do TQDM mostra:

* **Loss:** Erro total decrescente.
* **IoU:** Interseção sobre união global (precisão espacial).
* **G_V / G_I:** A média de confiança que o modelo está atribuindo ao Visível e ao Infravermelho, acompanhada do desvio padrão ( STD). Se o drone sumir no RGB, você verá o `G_V` cair e o `G_I` dominar.

---
