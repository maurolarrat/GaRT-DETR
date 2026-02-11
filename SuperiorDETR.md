* **o que o SuperiorDETR é**
* **de onde vêm as ideias**
* **quais modelos influenciaram cada parte**
* **o que foi adaptado**
* **como os dados fluem linha por linha**
* **por que cada escolha existe**

---

# SuperiorDETR: Multimodal Temporal RGBT Tracking Transformer

##  Visão Geral

O **SuperiorDETR** é um modelo de *tracking* multimodal RGBT (RGB + Infravermelho Térmico) de última geração, baseado inteiramente em Transformers. Ele foi projetado especificamente para o desafio de **rastrear alvos pequenos e rápidos (como drones)** em cenários onde a iluminação falha ou o contraste térmico é baixo.

Diferente de arquiteturas de rastreamento convencionais, o SuperiorDETR elimina a necessidade de:

* **Filtros de Kalman** (o estado é mantido pelas queries).
* **NMS (Non-Maximum Suppression)** (o Transformer aprende a não duplicar detecções).
* **Associação Heurística de Dados** (a identidade é preservada temporalmente).

---

##  Influências e Linhagem Científica

O SuperiorDETR não reinventa a roda; ele combina o "estado da arte" de várias linhagens de modelos:

| Componente | Influência Principal | O que foi adaptado |
| --- | --- | --- |
| **Arquitetura de Base** | **DETR (Facebook AI)** | Uso de *object queries* e regressão direta de caixas sem *anchors*. |
| **Refinamento de Caixa** | **Deformable-DETR** | Em vez de prever a caixa do zero, o modelo prevê um  (delta) iterativo sobre uma caixa de referência. |
| **Foco Espacial** | **Sparse R-CNN** | O conceito de *Refinement Layers* que focam apenas em áreas de interesse. |
| **Rastreamento** | **TrackFormer / TransTrack** | Propagação de queries entre frames para manter a identidade do alvo. |
| **Fusão Multimodal** | **Gated Multimodal Units** | Mecanismo de *Gating* para decidir dinamicamente se o RGB ou o IR é mais confiável no momento. |

---

##  Detalhamento da Arquitetura (Linha por Linha)

### 1. Backbone RGBT com Fusão Simétrica

O modelo utiliza duas **ResNet-18** independentes. A grande sacada aqui é a **fusão simétrica**: o IR não é apenas um "extra", ele ativamente limpa o ruído do RGB e vice-versa através do `GatedFusionBlock`.

* **Gating Dinâmico:** O modelo calcula um score de confiança ( a ) para cada modalidade. Se o drone entrar em uma zona de sombra, o gate do RGB fecha e o do IR abre automaticamente.
* **Bottleneck:** Reduz a dimensão concatenada () de volta para  para manter a eficiência computacional.

### 2. RefinementLayer: O "Coração" do Modelo

Localizada na classe `RefinementLayer`, esta camada substitui o decodificador padrão do Transformer por algo mais cirúrgico:

* **Soft ROI-Attention:** Usamos um viés Gaussiano (`attn_bias`) baseado na distância euclidiana entre a query e o mapa de características.



Isso força a atenção a ignorar o resto da imagem e focar apenas no entorno da caixa de referência.
* **Sigma Dinâmico:** Conforme as camadas avançam, o  diminui. Ou seja, a atenção fica mais "focada" e exigente à medida que o modelo ganha certeza da localização.

### 3. High-Res Zoom (Amostragem Local)

Na camada 3 do decodificador, o modelo realiza um `F.grid_sample`. Ele volta nas características de alta resolução do backbone () e extrai um patch específico de onde ele acha que o drone está.

* **Por que?** Detectar um drone de 10 pixels em uma imagem de 224 pixels é difícil. O Zoom local recupera os detalhes espaciais perdidos no *downsampling* do backbone.

### 4. Identidade Temporal Suave

A implementação da **Melhoria 1** evita o "flicker" (piscar) do alvo:

```python
alpha = torch.sigmoid(last_exist)
Q_t = Q_t * alpha + self.query_embed.weight * (1.0 - alpha)

```

Se o modelo tem  de certeza que o drone existia no frame anterior, ele mantém  da "memória" daquela query e apenas  da query genérica. Isso cria um rastreamento fluido e estável.

---

##  Fluxo de Dados (Data Flow)

1. **Entrada:** Recebe tensores de imagens Visíveis e Infravermelhas .
2. **Backbone:** As imagens passam pelas ResNets; o `GatedFusionBlock` mistura as modalidades gerando o `memory_all`.
3. **Encoder:** Um Transformer Encoder espacial limpa as dependências globais de cada frame.
4. **Temporal Loop:**
* As queries do frame anterior são atualizadas pela confiança de existência.
* Passam por 6 camadas de refinamento.
* Em cada camada, a caixa de referência é atualizada: `ref_t = sigmoid(logit(ref_t) + delta)`.


5. **Cabeças de Saída:**
* `pred_boxes`: Coordenadas  normalizadas.
* `exist`: Probabilidade global de o objeto estar presente (máximo entre Vis, IR e Global).



---

##  Por que estas escolhas existem?

* **Uso do `tanh() * 0.2` na BBox:** Limita o quanto a caixa pode "pular" em uma única camada, evitando instabilidade numérica no treino.
* **Trabalhar no espaço Logit:** Realizar somas em logit antes do `sigmoid` garante que a geometria da caixa seja preservada de forma linear, facilitando a convergência do gradiente.
* **Padding Count no Dataloader:** Permite lidar com sequências de vídeo de tamanhos diferentes dentro do mesmo batch, essencial para o dataset Anti-UAV.

---

*Este documento reflete a implementação exata contida no arquivo `SuperiorDETR.py`.*

---
