# Documentação Técnica: Pipeline de Dados e Integração de Perda AntiUAV-RGBT

Este documento detalha a arquitetura do DataLoader e sua sinergia com o critério de otimização para o rastreamento multimodal de drones (Visível e Infravermelho).

---

## 1. Arquitetura da Classe `AntiUAVRGBTDataset`

A classe implementa um fluxo de dados baseado em **sequências temporais**, essencial para modelos que exploram a continuidade do movimento e a correlação entre sensores.

### 1.1. Gerenciamento de Memória e I/O
No método `__init__`, o pipeline executa o *parsing* antecipado de todos os arquivos JSON.
* **Anotação em Cache:** Os metadados são armazenados em `self.annotation_cache` (RAM), eliminando gargalos de I/O durante o treinamento.
* **Filtro de Integridade:** Garante que apenas frames com anotações completas $[x, y, w, h]$ em ambas as modalidades sejam processados.

### 1.2. Estratégia de Amostragem Temporal
A extração de dados via `__getitem__` utiliza uma **Janela Temporal Adaptativa**:
* **Temporal Window ($T$):** Define a profundidade da série temporal. Se a sequência for menor que $T$, aplica-se *padding* por repetição.
* **Amostragem Aleatória:** Funciona como um aumento de dados temporal, permitindo que o modelo veja diferentes trechos do mesmo vídeo em épocas distintas.

---

## 2. Processamento Multimodal (Early Fusion)

O pipeline funde as modalidades no nível de entrada, criando um tensor de 4 canais.

### 2.1. Fusão de Canais
A concatenação ocorre na dimensão dos canais:
$$X_{input} \in \mathbb{R}^{B \times T \times 4 \times H \times W}$$
Onde os 4 canais representam $\{R, G, B, IR\}$.

* **Implicação Acadêmica:** Esta configuração caracteriza uma **Fusão Precoce**. Permite que o backbone aprenda filtros espaciais que correlacionam a assinatura térmica com as características visuais desde as camadas mais rasas.

### 2.2. Normalização de Coordenadas
As *Bounding Boxes* são convertidas de pixels para o intervalo $[0, 1]$ no formato $CXCYWH$:
$$CX = \frac{x + w/2}{Width_{image}}, \quad CY = \frac{y + h/2}{Height_{image}}$$
$$W = \frac{w}{Width_{image}}, \quad H = \frac{h}{Height_{image}}$$

---

## 3. Sinergia com Funções de Perda (Loss Functions)

A estrutura de dados possui implicações diretas na estabilidade numérica e na convergência do otimizador.

### 3.1. Regressão de Bounding Boxes
O modelo compara as predições com os alvos `boxes_vis` e `boxes_ir`:
* **L1 Loss (Erro Absoluto):** A normalização garante que erros em frames de diferentes resoluções tenham o mesmo peso, evitando que sequências de alta resolução dominem o gradiente.
* **Generalized IoU (GIoU) Loss:** Calculada de forma invariante à escala, facilitada pelo formato centralizado das bboxes.

### 3.2. Supervisão de Existência
O campo `exist` (tensor binário) atua como a verdade fundamental para a cabeça de classificação.
* **Critério Rigoroso:** O drone só "existe" se detectado em **ambas** as câmeras. Isso penaliza o modelo caso ele confie em sensores isolados que apresentam ruído (ex: reflexos no visível ou calor residual no IR).

---

## 4. Fluxo de Tensores: Do Dataset ao Critério

A tabela abaixo descreve a evolução dimensional dos dados no pipeline:

| Etapa | Tensor / Operação | Dimensão (Shape) | Função no Pipeline |
| :--- | :--- | :--- | :--- |
| **Input** | `x_input` | $[B, T, 4, H, W]$ | Entrada multimodal (Early Fusion). |
| **GT Box** | `boxes_vis` / `boxes_ir` | $[B, T, 4]$ | Alvos para regressão geométrica. |
| **GT Exist**| `exist` | $[B, T]$ | Alvo para classificação de presença. |
| **Output** | `pred_boxes` | $[B, T, Q, 4]$ | Predições do modelo ($Q$ queries). |

---

## 5. Considerações sobre o "Tau Adaptativo" na Avaliação

Diferente do treino (binário), a validação utiliza a técnica de **Soft Accuracy (SA)**:
* **Métrica mSA:** A acurácia é ponderada pela confiança suavizada do modelo através de um limiar adaptativo $\tau$.
* **Cálculo:** Se $Exist_{gt} = 1$, a pontuação é $Confiança \times IoU$. Se $Exist_{gt} = 0$, a pontuação é $1 - Confiança$.
* **Dependência:** Esta métrica depende da limpeza de dados realizada no `__init__`, garantindo que a penalização seja aplicada apenas sobre frames com anotações confiáveis.

---

## 6. Sinergia entre Dataloader e Funções de Perda (Loss Functions)

A estrutura de dados fornecida pelo `AntiUAVRGBTDataset` — especificamente a normalização para o formato $CXCYWH$ no intervalo $[0, 1]$ — possui implicações diretas na estabilidade numérica e na convergência do otimizador durante o cálculo do critério de perda.

### 6.1. Regressão de Bounding Boxes: L1 e Generalized IoU (GIoU)
O modelo prediz as coordenadas das caixas, que são comparadas aos alvos `boxes_vis` e `boxes_ir` gerados pelo Dataloader.

* **L1 Loss (Erro Absoluto):** Atua diretamente sobre as coordenadas normalizadas. 
    * **Implicação:** A normalização no intervalo $[0, 1]$ garante que erros em frames de alta resolução (ex: Full HD) tenham o mesmo peso que erros em frames de baixa resolução. Isso previne que sequências com resoluções maiores dominem o gradiente e enviesem o aprendizado.
* **GIoU Loss (Generalized Intersection over Union):** Como as bboxes são entregues pelo Dataloader no formato centralizado ($CX, CY, W, H$), a função de custo pode calcular a intersecção sobre a união de forma invariante à escala.
    * **Vantagem:** O GIoU resolve o problema de gradientes nulos quando não há sobreposição entre a predição e o alvo, algo crítico em alvos pequenos como drones.



### 6.2. Supervisão de Existência e Classificação Binária
O campo `exist` (tensor binário) gerado no `__getitem__` atua como o *Ground Truth* para a cabeça de classificação de existência do drone.

* **Lógica de Consistência Multimodal:** O Dataloader define a existência ($exist = 1$) apenas quando o drone está presente em **ambas** as modalidades simultaneamente.
* **Implicação no Treino:** O modelo é penalizado via *Cross Entropy* ou *Binary Cross Entropy* se confiar excessivamente em apenas um sensor. Isso obriga a rede a aprender uma representação robusta, ignorando falso-positivos comuns como reflexos solares no canal Visível ou fontes de calor irrelevantes (clutter) no canal Infravermelho.

---

> **Destaque para a Revisão:**
> Esta arquitetura de dados foi projetada para ser agnóstica à resolução original dos sensores. A fusão precoce (4 canais) assegura que o backbone convolucional extraia features correlacionadas desde o primeiro nível de abstração espacial, otimizando a detecção em ambientes complexos.
