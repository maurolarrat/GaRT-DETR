# Documentação Técnica: Modelo GaRT-DETR (RGBT Tracking)

Este documento descreve a arquitetura e o fluxo de dados do modelo **GaRT-DETR**, um detector/rastreador multimodal que combina informações de sensores Visíveis (RGB) e Infravermelhos (IR) utilizando mecanismos de atenção espacial e refinamento iterativo.

## 1. Pré-processamento Temporal (`preprocess_batch`)

O pipeline começa com o tratamento dos dados de entrada. Como o modelo lida com vídeos (sequências), esta função organiza o "caos" dimensional:

* **Flatten Temporal:** Concatena todos os frames de todas as sequências do batch em um único tensor gigante () para processamento eficiente no backbone.
* **Redimensionamento:** Garante que ambos os sensores (RGB e IR) estejam na mesma resolução alvo (ex: 224x224).
* **Preservação de Metadados:** Armazena os tamanhos originais das imagens para que, no final, as caixas delimitadoras possam ser convertidas de volta para os pixels reais do vídeo original.

## 2. Blocos de Fusão com Gating

O modelo utiliza dois tipos de blocos para fundir as informações dos sensores, decidindo "o quanto" confiar em cada um:

### `GatedFusionBlock` (Fusão Global)

* **Mecânica:** Usa Atenção Cruzada (*Cross-Attention*) para que o braço principal consulte o braço auxiliar.
* **Gate de Confiança:** Calcula um valor escalar único para a imagem inteira. Se o IR estiver ruidoso, o gate fecha, diminuindo a influência do IR no RGB.

### `SpatialGatedFusionBlock` (Fusão Espacial)

* **Mecânica:** Diferente do anterior, este calcula a confiança **por região** (token).
* **Foco:** Se houver fumaça em apenas uma parte do frame RGB, o modelo pode escolher confiar no IR apenas naquela região específica, mantendo o RGB para o restante da imagem.

## 3. O Backbone Multimodal (`RGBTBackbone`)

Este é o extrator de características de "dois braços":

* **Braço RGB (ResNet18):** Especialista em capturar texturas, cores e bordas finas.
* **Braço IR (EfficientNet-B0):** Adaptado cirurgicamente para aceitar 1 canal (térmico). Ele foca em assinaturas de calor e formas que persistem em baixa luminosidade.
* **Extração Multinível:** O backbone extrai tanto *features* profundas (para semântica) quanto de alta resolução (para localização precisa).
* **Memória Multimodal:** O resultado final é uma representação fundida que serve como a "memória visual" para o Transformer.

## 4. Camada de Refinamento (`RefinementLayer`)

Este bloco implementa a **Soft ROI-Attention**, uma das principais inovações do código:

* **Auto-Atenção:** As queries (propostas de drones) conversam entre si para evitar detecções duplicadas.
* **Atenção Gaussiana:** Em vez de olhar para a imagem toda, a atenção é multiplicada por um "bias" que força o modelo a olhar apenas ao redor da posição atual da query.
* **Foco Progressivo:** À medida que passamos pelas camadas (0 a 6), o raio de visão (Sigma) diminui, forçando o modelo a ser cada vez mais preciso.

## 5. Arquitetura Central (`GaRT-DETR`)

A classe mestre que orquestra o fluxo temporal e iterativo:

### Inicialização de Queries

* O modelo não começa do zero. Ele inicializa um **Grid Proporcional** de pontos de referência distribuídos pela imagem, garantindo que nenhuma região seja ignorada no início.

### Propagação Temporal (Smooth Tracking)

* **Mecanismo de Alpha-Blending:** O modelo usa a confiança do frame anterior para guiar o atual. Se o drone foi detectado com firmeza no frame `t-1`, a query no frame `t` começará exatamente naquela posição, criando um rastreio fluido e estável.

### Mecanismo de Zoom (High-Res Zoom)

* No meio do processo de refinamento (camada 2), o modelo faz uma pausa para olhar os detalhes.
* **`_make_sampling_grid`:** Gera coordenadas para "recortar" um patch de alta resolução de onde o objeto parece estar.
* **ROI Align Manual:** Usando `grid_sample`, ele extrai detalhes finos que o backbone profundo perdeu, reintegrando essa informação na query.

### Predição Desacoplada

* O modelo gera duas caixas: uma para o Visível e outra para o IR. Isso permite lidar com o efeito de **Paralaxe** (quando os sensores estão em posições físicas diferentes) ou quando o drone está visível em um sensor, mas ocluso em outro.

## 6. Motor de Amostragem (`_make_sampling_grid`)

Uma função utilitária geométrica que:

1. Pega as coordenadas `[0, 1]` da caixa.
2. Converte para o espaço de coordenadas do PyTorch `[-1, 1]`.
3. Aplica uma escala de **1.5x** para garantir que o recorte tenha um pouco de contexto ao redor do objeto.
4. Cria a malha de amostragem necessária para a função `F.grid_sample`.

---

### Resumo do Fluxo de Dados

1. **Entrada:** Batch de sequências RGBT.
2. **Backbone:** Fusão inteligente de sensores com gates de confiança.
3. **Transformer Encoder:** Refinamento global da memória.
4. **Loop Temporal:** Queries são propagadas de frame em frame com suavização.
5. **Refinamento Iterativo:** Queries buscam o drone na memória usando atenção focada e zoom local.
6. **Saída:** Coordenadas e scores de existência para ambos os sensores.
