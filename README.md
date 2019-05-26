# SER_347_Trabalho_Final - Luis Ricardo Arantes Filho

## Introdução
Este trabalho apresenta uma proposta de análise de eventos extremos em imagens de satélite, 
especificamente a caracterização de eventos climáticos como chuvas torrenciais, tempestades, furacões e ciclones. 
A classificação de eventos climáticos é de certa forma, algo de grande importância para todas as categorias sociais 
e governamentais.

Eventos climáticos, principalmente os considerados extremos, como ciclones, furacões e tsunamis, são determinantes no comportamento
econômico e social, pois acarretam em grandes prejuízos. As mudanças climáticas são um fator determinante para o 
aumento da frequência de eventos climáticos extremos, identificar e emitir alertas em um curto período de tempo para tomada de 
decisões é um desafio.

A análise destes eventos pode ser feita de maneira visual indicando nas imagens de satélites regiões propensas a ocorrência de eventos extremos. Entretanto, avaliar um grande volume de dados gerados diariamente é uma tarefa não-trivial, ainda mais, quando é necessária a avaliação de especialistas para indicar a ocorrência destes eventos.


## Objetivos Gerais
Esta proposta apresenta um modelo baseado em Redes Neurais Convolutivas para aprendizagem dos padrões em imagens de satélite de forma a identificar eventos extremos como ciclones e furacões.

## Materiais e Métodos

### Redes Neurais Profundas

As motivações para criação de arquiteturas profundas e Redes Neurais Profundas, Deep Neural Networks (DNN), 
provém da criação de modelos e algoritmos estruturados em muitas camadas de processamento em que não exista prejuízo 
a generalização e ao aprendizado de máquina.

A inclusão de mais camadas intermediárias em Redes Neurais Artificiais (RNAs) vem da constatação 
de que o processamento de informações não lineares e o mapeamento das funções torna-se mais eficiente. 
Conforme [REF1] RNAs com uma camada intermediária conseguem implementar qualquer tipo de função contínua, 
enquanto que, duas camadas permitem a aproximação de qualquer tipo de função matemática. 
Em redes MLP esta aproximação nem sempre converge para uma solução ótima, pois a função de mapeamento depende fortemente da 
distribuição inicial dos dados [REF2]. Mesmo podendo aproximar qualquer função 
o aprofundamento destas redes pode gerar uma solução que converge para mínimos e máximos locais.

O aprofundamento das RNAs permite o mapeamento das características dos dados gerando uma codificação 
interna para o conjunto treinado. Desta forma quanto maior o número de camadas mais refinada é abstração de características.
Esta definição não garante um aprendizado ótimo já que a adição de mais camadas pode gerar uma 
generalização rígida (*overfitting*). Além disso quanto maior for a dimensionalidade dos dados maior 
será a quantidade de camadas para descrever estes dados. Em certos pontos, o número de camadas cresce exponencialmente à 
medida que os dados tornam-se mais complexos.

### Redes Neurais Convolutivas

Redes Neurais Convolutivas ou Redes Convolucionais, *Convolutional Neural Networks* (CNNs) são RNAs que possuem uma 
arquitetura profunda e hierárquica, ou seja, redes CNNs tem a capacidade de abstrair informações dos dados brutos 
e representá-las em muitos níveis de abstração, isto é, das representações das características mais simples às mais complexas. 
Este tipo de rede é aplicável comumente a problemas de classificação de imagens, reconhecimento de objetos e demais problemas
relacionados à área de visão computacional.

As primeiras redes com o conceito de arquiteturas profundas e com operações de convolução foram propostas por 
[REF3,REF4], e foram denominadas LeNets, as redes LeNets foram desenvolvidas para o reconhecimento de padrões em imagens. 
A Figura 1 ilustra uma arquitetura simples da rede LeNet para classificação de caracteres.


<p align="center"> Figura 1 - Rede Convolutiva proposta por Lecun</p>

<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/lecun.png">
</p>
<p align="center"> Fonte: REF3 </p>
