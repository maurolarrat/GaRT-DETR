Os detalhes deste trabalho estão descritos no artigo compartilhado no Overleaf.

# Proposta de Tese para a Qualificação

## 1. Título
**"Rastreamento de Micro-Alvos Multimodais via Transformadores de Atenção Localizada e Fusão Espacial Baseada em Incerteza Aleatória Aprendida"**

## 2. Motivação e Justificativa
O rastreamento de micro-veículos aéreos não tripulados (**micro-UAVs**) apresenta desafios críticos devido à baixa resolução dos alvos (muitas vezes ocupando menos de 0.5% da imagem), manobras erráticas e condições ambientais adversas como neblina, oclusões e saturação térmica.

A motivação deste trabalho reside na insuficiência de algoritmos de rastreamento de modalidade única (apenas RGB ou apenas IR) em cenários complexos. Justifica-se a necessidade de um sistema robusto que não apenas funda dados multiespectrais, mas que **aprenda a confiabilidade dinâmica** de cada sensor através da incerteza intrínseca, evitando que uma modalidade degradada prejudique a predição final.

## 3. Estado da Arte
Conforme definido na **Revisão Sistemática de Literatura (RSL)** prévia (referenciar o artigo da RSL que a ACM está há 3375 anos revisando...), o estado da arte em rastreamento RGBT evoluiu de fusões no nível de pixels para arquiteturas baseadas em transformadores (ViT, DETR). No entanto, persistem lacunas em:

* **Seleção de Características:** A maioria dos modelos utiliza pesos estáticos ou mecanismos de atenção global que ignoram a variância espacial do ruído.
* **Consistência Temporal:** O uso de filtros geométricos (como Filtro de Kalman) falha sob acelerações extremas e trajetórias não lineares de micro-drones.

## 4. O Problema Científico (A Tese)
A tese ataca a degradação da confiabilidade sensorial em ambientes dinâmicos através da modelagem de incerteza aleatória.

### Proposição Central:
> "Diferente de abordagens que utilizam pesos de fusão estáticos, a robustez no rastreamento de micro-UAVs é alcançada via gating atencional condicionado pela incerteza intrínseca (aleatória) de cada sensor. A integração de memória temporal residual e refinamento espacial 'Soft-ROI' permite manter a consistência da identidade do alvo mesmo sob falha catastrófica de um dos sensores."

## 5. Hipóteses de Pesquisa
* **H1 (Fusão por Incerteza):** A parametrização do bias de fusão como um logaritmo de variância aprendida (Incerteza de Kendall) permite uma 'ablação dinâmica' em tempo de execução, superando mecanismos que propagam ruído para o espaço latente multimodal.
* **H2 (Localização Soft-ROI):** A imposição de um bias Gaussiano decrescente nas *Refinement Layers* mitiga o ruído de fundo (*clutter*) em micro-objetos, forçando a convergência da rede para gradientes locais sem a rigidez matemática do ROI Align tradicional.

## 6. Objetivos

### Objetivo Geral
Desenvolver e validar uma arquitetura de rastreamento multimodal baseada em transformadores capaz de operar de forma resiliente em vídeos RGBT de micro-alvos.

### Objetivos Específicos
1.  Implementar um *backbone* RGBT que utilize **Modality Dropout** para garantir aprendizado ortogonal e robustez a falhas de sensores.
2.  Criar um mecanismo de **Gating Espacial** (via `SpatialGatedFusionBlock`) para o canal infravermelho, permitindo a filtragem seletiva de ruído térmico.
3.  Desenvolver uma camada de **Refinamento Iterativo** com atenção ROI-Soft para lidar com a extrema escala reduzida dos alvos.
4.  Avaliar a eficácia da **Inércia de Queries** na manutenção da consistência temporal frente a manobras erráticas.

## 7. Metodologia Proposta (Contribuições Arquiteturais)



### I. O Paradigma da Fusão Assimétrica
No modelo proposto, o braço visível utiliza um gate global, enquanto o infravermelho aplica um `SpatialGatedFusionBlock`. Dado que o sensor térmico é frequentemente mais ruidoso, mas semanticamente mais simples, o gate espacial permite que o IR contribua com "manchas de calor" localizadas, enquanto o RGB fornece a estrutura global.

### II. Propagação Temporal via Inércia de Queries
A query do frame $t$ é uma interpolação entre o embedding aprendido e o estado do frame $t-1$, ponderada pela confiança combinada. Isto substitui modelos físicos rígidos por uma **inércia latente no espaço de busca**, garantindo que a memória semântica do drone guie a atenção no frame subsequente.

### III. Aprendizado sob Dropout de Modalidade (Indiscriminate Learning)
O *dataloader* e o *backbone* são projetados para o treinamento indiscriminado: o modelo é exposto a apenas um sensor em 50% das iterações (25% VIS-only, 25% IR-only). Isso força a rede a desenvolver redundância latente, essencial para operação em oclusão total ou ausência de luz.

## 8. Resultados Esperados
A pesquisa propõe o **Superior-DETR**, uma arquitetura *end-to-end*. A contribuição original reside na **Fusão Cruzada por Gating de Incerteza**, permitindo adaptação dinâmica a cenários de neblina ou saturação. Espera-se demonstrar que o desacoplamento das cabeças de predição, aliado ao recurso de "Zoom" de alta resolução (`grid_sample`), minimiza o erro de *drift* em micro-escalas, superando o estado da arte em métricas de sucesso temporal (Reset-Lock).

---
