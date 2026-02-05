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
