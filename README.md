# IndisNET
Indiscriminate Network for Anti-UAV

Esta documentação técnica foi estruturada para refletir um rigor acadêmico adequado para avaliações de especialistas, focando na arquitetura de dados e nas implicações de design do pipeline de pré-processamento para rastreamento de alvos multimodais (RGBT).

Documentação Técnica: Pipeline de Dados AntiUAVRGBT
O pipeline implementado através da classe AntiUAVRGBTDataset e da função collate_fn_superior estabelece um fluxo de dados robusto para o treinamento de modelos de aprendizado profundo aplicados ao rastreamento e detecção de veículos aéreos não tripulados (UAVs) em espectro dual.
1. Arquitetura da Classe AntiUAVRGBTDataset
A classe herda de torch.utils.data.Dataset, adotando uma estratégia de carregamento baseada em sequências temporais em vez de frames isolados, o que é fundamental para modelos que exploram a correlação temporal.
1.1. Estratégia de Cache e Gerenciamento de Memória
No método __init__, o pipeline executa o parsing antecipado de todos os arquivos JSON de anotação.
 * Implicação Técnica: Esta abordagem elimina o gargalo de I/O de disco referente à leitura de metadados durante o loop de treinamento. Ao armazenar as coordenadas de Ground Truth (GT) e flags de existência na RAM (self.annotation_cache), otimiza-se o throughput da GPU, mantendo-a ocupada com o processamento de tensores de imagem.
1.2. Filtro de Integridade Multimodal
O código implementa um filtro determinístico para identificar valid_indices:
valid_indices = [i for i in range(len(v_data["gt_rect"])) if len(v_data["gt_rect"][i]) == 4 ...]

 * Funcionalidade: Garante que apenas frames com anotações de bboxes completas em ambas as modalidades (Visível e Infravermelho) sejam considerados.
 * Implicação: Previne falhas catastróficas durante o cálculo de perda (Loss) e assegura que a fusão multimodal ocorra sobre dados semanticamente alinhados.
2. Dinâmica Temporal e Amostragem
O método __getitem__ é o núcleo operacional da extração de dados.
2.1. Janela Temporal Adaptativa (temporal_window)
O parâmetro temporal_window define a profundidade da série temporal enviada ao modelo.
 * Amostragem Aleatória: Para sequências longas, um ponto de partida aleatório é selecionado, servindo como uma forma de Data Augmentation temporal.
 * Tratamento de Exceções (Padding): Caso a sequência seja inferior à janela, o pipeline aplica um padding de repetição do último frame válido. Isso mantém a dimensionalidade do tensor estática, requisito necessário para o processamento em lote (batch processing).
2.2. Fusão de Canais de Entrada
A fusão ocorre na dimensão dos canais:
"x_input": torch.cat([torch.stack(vis_tensors), torch.stack(ir_tensors)], dim=1)

 * Configuração de Saída: Resulta em um tensor de entrada com 4 canais (R, G, B, IR).
 * Implicação Acadêmica: Esta configuração caracteriza uma estratégia de Fusão Precoce (Early Fusion). O modelo recebe a informação térmica como um canal extra de textura/intensidade, permitindo que as primeiras camadas convolucionais aprendam filtros espaciais que correlacionam luz visível e calor simultaneamente.
3. Normalização e Geometria de Bounding Boxes
O pipeline realiza a conversão sistemática de coordenadas absolutas (pixels) para coordenadas relativas [0, 1].
| Parâmetro | Transformação Realizada | Objetivo |
|---|---|---|
| Escalonamento | (x, y, w, h) / (W, H) | Independência de resolução de entrada. |
| Centroide | x + (w/2) | Conversão para formato CXCYWH. |
| Consistência | exist_vis AND exist_ir | Definição de presença real do alvo no par multimodal. |
 * Nota de Design: O uso do formato CXCYWH (Center X, Center Y, Width, Height) é o padrão para arquiteturas modernas como DETR ou YOLO, facilitando a convergência da regressão de bboxes.
4. Agregação por collate_fn_superior
A função de colação personalizada é responsável por organizar a estrutura de tensores para o DataLoader.
 * Dimensionalidade Final: O tensor x_input entregue ao modelo possui o shape:
   
   
   Onde:
   * B: Batch size (Lote).
   * T: Temporal window (Janela temporal).
   * C: 4 (Canais RGB + IR).
   * H, W: Altura e largura definidas pelo transform.
5. Resumo de Implicações de Parâmetros
 * root_dir & split: Definem o escopo de generalização do modelo.
 * temporal_window: Controla a carga de memória da GPU. Valores altos aumentam a capacidade de modelar oclusões, mas exigem mais VRAM.
 * transform: Parâmetro crítico. Diferenças de interpolação entre as modalidades Visível e IR podem introduzir artefatos se não forem tratadas de forma homogênea.

6. Sinergia entre Dataloader e Funções de Perda (Loss Functions)
A estrutura de dados fornecida pelo AntiUAVRGBTDataset — especificamente a normalização para o formato CXCYWH no intervalo [0, 1] — possui implicações diretas na estabilidade numérica e na convergência do otimizador.
6.1. Regressão de Bounding Boxes: L1 e Generalized IoU (GIoU)
O modelo prediz as coordenadas das caixas, que são comparadas aos alvos boxes_vis e boxes_ir gerados pelo Dataloader.
 * L1 Loss (Erro Absoluto): Atua diretamente sobre as coordenadas normalizadas. A normalização [0, 1] garante que erros em frames de alta resolução (Full HD) tenham o mesmo peso que erros em frames de baixa resolução, prevenindo que sequências com resoluções maiores dominem o gradiente.
 * GIoU Loss: Como as bboxes estão no formato centralizado (CX, CY, W, H), a função de custo pode calcular a intersecção sobre a união de forma invariante à escala.
6.2. Supervisão de Existência e Classificação Binária
O campo exist (tensor binário) atua como o Ground Truth para a cabeça de classificação de existência do drone.
 * Implicação: Como o Dataloader define a existência apenas quando o drone está presente em ambas as modalidades, o modelo é penalizado se confiar excessivamente em apenas um sensor que pode estar apresentando falso-positivos (ex: reflexos solares no Visível ou fontes de calor irrelevantes no IR).
7. Fluxo de Tensores: Do Dataset ao Critério
O diagrama abaixo ilustra como as dimensões dos tensores evoluem desde a amostragem no disco até a aplicação das métricas de erro.
| Etapa | Tensor / Operação | Dimensão (Shape) | Função no Pipeline |
|---|---|---|---|
| Input | x_input | [B, T, 4, H, W] | Entrada multimodal (Early Fusion). |
| GT Box | boxes_vis / boxes_ir | [B, T, 4] | Alvos para regressão geométrica. |
| GT Exist | exist | [B, T] | Alvo para classificação de presença. |
| Output | pred_boxes | [B, T, Q, 4] | Predições do modelo (Q queries). |
8. Considerações sobre o "Tau Adaptativo" na Avaliação
Embora o Dataloader entregue um valor binário de existência, o pipeline de validação utiliza uma técnica de Soft Accuracy (SA) baseada na confiança do modelo.
 * Métrica mSA: Em vez de um corte rígido (Threshold), a acurácia é ponderada pela confiança suavizada. Se o exist_gt for 0, o modelo deve minimizar sua confiança; se for 1, a pontuação é o produto da confiança pelo IoU alcançado.
 * Conexão com o Dataset: Esta métrica depende intrinsecamente da limpeza de dados realizada no __init__, onde frames ruidosos foram previamente filtrados para garantir que a penalização do modelo seja justa.
Destaque para o Orientador:
Esta arquitetura de dados foi projetada para ser agnóstica à resolução original dos sensores, permitindo que o modelo aprenda características semânticas de alto nível. A fusão precoce (4 canais) assegura que o backbone convolucional extraia features correlacionadas desde o primeiro nível de abstração espacial.
