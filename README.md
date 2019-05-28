# Detecção de Eventos Climáticos Extremos em Imagens de Satélite Utilizando Redes Neurais Convolutivas
### SER_347_Trabalho_Final - Luis Ricardo Arantes Filho

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

As redes neurais convolutivas são compostas por uma sequência de camadas com funções de processamento específicas:

- Camada de entrada: composta por uma matriz multidimensional, descrevendo os dados. Por exemplo: Vetores de
características, imagens, sinais, etc.

- Camada convolucional: esta camada é responsável pelas operações de convolução e processamento de características. De maneira simples, a camada de convolução é responsável por mapear as características principais dos dados de entrada.

- Camada de Pooling: esta camada é responsável por sumarizar o processamento feito nas camadas de convolução, esta operação abstrai do resultante da convolução as características mais evidentes, reduzindo assim, a dimensionalidade dos dados.
    
- Camada Totalmente Conectada: esta camada se comporta como uma camada de processamento de redes neurais clássicas, como a rede Multilayer Perceptron. Nesta camada são inseridos os parâmetros resultantes das camadas anteriores e o treinamento do modelo é feito pelo algoritmo de retro-propagação do erro (*backpropagation*)


O treinamento das redes CNNs ocorre de forma similar ao modelo clássico de RNAs, quando se atinge as camadas totalmente conectadas. Cada uma das camadas são submetidas a um processamento não-linear executado por uma função de ativação, como por exemplo, a função sigmoidal, ReLu, softmax, tangente hiperbólica, etc. Estas funções podem ser escolhidas de maneira personalizada de acordo com as necessidades de determinado problema.

Desta forma, a aplicação de redes CNNs para avaliação e classificação de objetos em imagens de satélite é justificável. 

## Dados utilizados e Principais Características
O acesso aos dados usados neste trabalho pode ser visto no seguinte notebook, que explica cada passo para o acesso e manipulaçã das informações. As figuras 2 e 3 representam o produto de dados utilizado nesta aplicação para os dados obtidos do satelite goes e do repositorio HURSAT, respectivamente. 

### [Manipulando dados brutos GOES E HURSAT](Dados_Acesso_HURSAT_GOES16.ipynb)

<p align="center"> Figura 2 - Produto de imagem de evento extremo GOES</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/goes.png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>

<p align="center"> Figura 3 - Produto de imagem de evento extremo HURSAT</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/hursat.png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>


## Solução

As imagens a serem utilizadas no treinamento das redes CNNs são caracterizadas por dados de eventos extremos, como ciclones, tornados e furacões. As imagens obtidas neste trabalho são divididas em imagens com eventos e sem eventos extremos, conforme ilustra a Figura 4. A Figura 5 ilustra o produto de imagens do HURSAT, que foram utilizadas para o treinamento dos modelos de eventos extremos.

<p align="center"> Figura 4 - Imagens com eventos e sem eventos</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/all-img-ev-sv.png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>

<p align="center"> Figura 5 - Imagens com eventos Hursat</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/all-img-ev-sv.png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>


Os notebooks a seguir descrevem todos os passos para construção do modelo de Deep Learning para construção do classificador de eventos extremos. A figura 6 ilustra como o processo do tratamento de dados é feito e como os dados são inseridos no modelo de rede neural convolutiva.

### [Etapa de Geração de Imagens do produto HURSAT](Manipulando_Hursat.ipynb)
### [Etapa de ajuste de dados e construção dos dataFrames](ajustaVetores.ipynb)
### [Classificador de Eventos Extremos Utilizando Redes Neurais Convolutivas](Rede_cnn_eventos_extremos.ipynb)

<p align="center"> Figura 6 - Classificador de Eventos Extremos Utilizando Redes Convolutivas</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/Fluxo_Eventos.png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>

A arquitetura da rede neural utilizada inspira-se no modelo de Lecun como critério de escolha de camadas. Neste trabalho buscamos adotar apenas a modificação das dimensões das camadas. A figura 7 ilustra a arquitetura utilizada.


<p align="center"> Figura 7 - Arquitetura do modelo de Redes Convolutivas</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/arquit.png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>

## Resultados e Discussão
Os resultados obtidos com a abordagem são descritos na Tab.1. Os testes foram feitos para 544 imagens sendo 50% para eventos e 50% para não-eventos.

<p align="center"> Tabela 1 - Resultados de Classificação de Eventos Extremos</p>

<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/tab.png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>

A Figura 8 indica o comportamento do modelo no treinamento e na validação do modelo. A curva revela que o modelo generaliza o conjunto de treinamento em quase 99% das amostras, para o conjunto de teste obtém-se 98,5% na eficiência da classificação.

<p align="center"> Figura 8 - Resultados do Treinamento e Validação</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/train.png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>

A classificação de eventos extremos obteve a acurácia de 99,84% no conjunto de
treinamento (66% de toda amostra de imagens) e 98,53% no conjunto de testes (33%
de toda a amostra de imagens), este desempenho ilustra um mapeamento que cobre
boa parte das características de eventos extremos sendo capaz de classificar com uma
acurácia de 98%, 597 imagens que não foram submetidas ao modelo

É possível observar uma anomalia durante o treinamento entre a época 60 e 80, em
que o erro do modelo aumenta significativamente, ou seja, o modelo erra todos os
exemplos de teste. Após esta anomalia o modelo retorna ao aprendizado diminuindo
ainda mais o erro obtido anteriormente.

Em relação às caracteristicas treinadas pela rede convolutiva as Figuras 9 e 10 ilustram como o modelo extraiu as principais caracteristicas do padrão de ciclone e furação para identificar os eventos extremos sem confundir com eventos considerados como alta concentração de nuvens e quando regioes sem nuvens são apresentadas ao classificador.

<p align="center"> Figura 9 - Padrao de Evento Extremo</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/pos (5).png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>

A figura 8 ilustra a extração do padrão de evento extremo para o treinamento do classificador.

<p align="center"> Figura 10 - Extração de Caracteristicas de Evento Extremo</p>
<p align="center">
<img src="https://github.com/LuisRicardoAF/SER_347_Trabalho_Final/blob/master/pos (3).png">
</p>
<p align="center"> Fonte: Produção do Autor. </p>

## Referências
[REF1] G Cybenko.  Continuous valued neural networks with two hidden layers are
sufficient, department of computer science.
Trfts. University, 1988.

[REF2] Antônio  de  P ́adua  Braga,  Andr ́e  Carlos  Ponce  de  Leon  Ferreira,  and  Te-
resa Bernarda Ludermir. Redes neurais artificiais:  teoria e aplicacoes. LTC Editora Rio de Janeiro, Brazil:, 2007.

[REF3] Yann  LeCun,  Yoshua  Bengio,  et  al.   Convolutional  networks  for  images,
speech, and time series.The handbook of brain theory and neural networks, 3361(10):1995, 1995.

[REF4] Yann LeCun, L ́eon Bottou, Yoshua Bengio, Patrick Haffner, et al. Gradientbased learning applied to document recognition. Proceedings  of  the  IEEE, 86(11):2278–2324, 1998.
