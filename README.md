# Desafio 2 - Assistente Corporativo

## 🤖 Contexto

Você faz parte de uma equipe responsável pelo desenvolvimento de soluções de inteligência artificial para uma empresa fictícia.

A empresa deseja disponibilizar um assistente virtual capaz de responder às dúvidas dos colaboradores sobre as políticas internas de viagens corporativas e reembolsos.

Atualmente, essas informações estão disponíveis em documentos internos, e os colaboradores precisam consultá-los manualmente para esclarecer dúvidas sobre limites de despesas, prazos, autorizações e procedimentos.

A equipe deseja facilitar esse processo por meio de um assistente que permita fazer perguntas em linguagem natural e receber respostas fundamentadas nas informações disponíveis na documentação.

Seu papel neste desafio é atuar no desenvolvimento dessa solução. A partir dos documentos fornecidos, você deverá implementar um sistema de **Retrieval-Augmented Generation (RAG)**, utilizando **LangGraph** para organizar o fluxo de recuperação das informações e geração das respostas.

> **Importante:** a empresa e a política de viagens utilizadas neste exercício são fictícias. O conteúdo foi elaborado exclusivamente para fins educacionais.

## 📄 Dados disponíveis

A pasta `chunks/` contém os textos de uma política interna fictícia de viagens corporativas e reembolsos, já segmentados em arquivos individuais.

Cada arquivo representa um **chunk**. O nome do arquivo contém o identificador e o título do trecho, enquanto o conteúdo do arquivo contém **somente o texto original do chunk**, sem campos ou metadados adicionais.

**Não é necessário implementar a etapa de chunking.** Utilize os conteúdos exatamente como fornecidos nos arquivos. Considere o identificador e o título presentes nos nomes dos arquivos como metadados que podem ser extraídos durante a indexação.

## 🎯 Desafio

A empresa deseja disponibilizar um assistente capaz de responder às dúvidas dos colaboradores utilizando exclusivamente as informações presentes na política de viagens corporativas.

A partir dos dados fornecidos, desenvolva uma solução que contemple a indexação dos documentos, a recuperação das informações relevantes e a geração de respostas utilizando um modelo de linguagem.

Não existe uma única forma correta de implementar todas as etapas. Suas escolhas devem ser justificadas de acordo com o problema e com as características da solução desenvolvida.

### 1. Entendimento do problema

Analise o contexto apresentado e o objetivo do assistente.

Você deverá:

* explicar o que é RAG e como essa abordagem pode ser utilizada no cenário proposto;
* identificar o papel da indexação, da recuperação e da geração de respostas;
* explicar por que as respostas do assistente devem ser fundamentadas nos documentos disponibilizados;
* identificar as limitações de uma solução baseada exclusivamente na documentação fornecida.

### 2. Indexação dos documentos

Prepare os chunks para que possam ser utilizados na recuperação das informações.

Você deverá:

* selecionar um modelo de embedding;
* gerar os embeddings dos chunks disponibilizados;
* selecionar um banco de dados vetorial para armazenar os embeddings;
* armazenar os vetores juntamente com os textos originais e seus identificadores;
* garantir que seja possível recuperar o conteúdo original de cada chunk a partir dos resultados da busca.

Você poderá utilizar diferentes tecnologias para o armazenamento vetorial, como:

* PostgreSQL com pgvector;
* Pinecone;
* ChromaDB;
* FAISS;
* Qdrant.

A escolha da tecnologia fica a seu critério. Explique os motivos que levaram à sua escolha.

### 3. Recuperação das informações

Implemente o mecanismo responsável por recuperar os trechos mais relevantes para cada pergunta enviada pelo usuário.

O fluxo deverá contemplar:

* receber a pergunta do usuário;
* gerar o embedding da pergunta utilizando o modelo selecionado;
* realizar uma busca semântica no banco de dados vetorial;
* recuperar os chunks mais relevantes;
* permitir configurar a quantidade de chunks retornados pela busca (Top-K).

Você também poderá implementar uma etapa de **reranking**, responsável por reordenar os chunks recuperados de acordo com sua relevância para a pergunta.

### 4. Geração das respostas

Implemente a etapa responsável por gerar respostas utilizando um modelo de linguagem (LLM).

Você deverá:

* construir um contexto a partir dos chunks recuperados;
* enviar a pergunta e o contexto para o LLM;
* orientar o modelo a responder com base exclusivamente nas informações recuperadas;
* retornar uma resposta clara e coerente com o conteúdo da documentação.

**Regra fundamental:** o assistente não deverá inventar informações. Caso os documentos recuperados não contenham informações suficientes para responder à pergunta, deverá informar que não encontrou informações na documentação disponibilizada.

### 5. Construção do fluxo com LangGraph

Organize o processo de recuperação e geração utilizando **LangGraph**.

Uma estrutura inicial sugerida é:

* `retrieve`: recupera os chunks relevantes no banco vetorial;
* `rerank`: reordena os documentos recuperados (opcional);
* `generate`: gera a resposta utilizando o contexto recuperado.

Você poderá expandir o grafo com outros nós, como avaliação da relevância dos documentos recuperados ou verificação da resposta gerada.

Explique a responsabilidade de cada etapa e como os dados são compartilhados ao longo do fluxo.

### 6. Testes e avaliação

Teste o assistente utilizando os exemplos de perguntas apresentados neste README.

Sua avaliação deverá:

* verificar se os chunks relevantes foram recuperados;
* avaliar se as respostas estão de acordo com o conteúdo da documentação;
* testar perguntas que exigem a combinação de informações de diferentes chunks;
* verificar o comportamento do assistente quando a resposta não está disponível na documentação;
* identificar limitações e oportunidades de melhoria.

Utilize os resultados dos testes para justificar suas conclusões sobre a solução desenvolvida.

## 🧪 Exemplos de perguntas e respostas esperadas

Os exemplos a seguir deverão ser utilizados para testar o sistema desenvolvido. Eles representam diferentes situações de consulta à política de viagens corporativas.

### 1. Pergunta direta

**Pergunta:**

Qual é o limite diário de reembolso para despesas com alimentação?

**Resposta esperada:**

O limite de reembolso para despesas com alimentação é de R$ 120,00 por dia de viagem. Esse valor contempla café da manhã, almoço e jantar. Despesas com bebidas alcoólicas não são reembolsáveis.

**Chunk esperado:** 006.

### 2. Pergunta que exige combinação de informações

**Pergunta:**

Se eu retornar de uma viagem na segunda-feira, até quando preciso apresentar a prestação de contas?

**Resposta esperada:**

A prestação de contas deverá ser apresentada em até 5 dias úteis após o retorno da viagem. Portanto, considerando o retorno na segunda-feira e desconsiderando feriados, o prazo termina na segunda-feira seguinte.

**Chunk esperado:** 010.

### 3. Pergunta que exige recuperação de múltiplos chunks

**Pergunta:**

Quais são os limites financeiros para hospedagem e alimentação durante uma viagem corporativa?

**Resposta esperada:**

Os limites estabelecidos são:

* Hospedagem: R$ 350,00 por diária, incluindo taxas obrigatórias.
* Alimentação: R$ 120,00 por dia de viagem, contemplando café da manhã, almoço e jantar.

Despesas de hospedagem acima do limite deverão ser previamente autorizadas pela diretoria responsável. Bebidas alcoólicas não são reembolsáveis.

**Chunks esperados:** 005 e 006.

### 4. Pergunta que exige interpretação de regras

**Pergunta:**

Posso comprar uma passagem aérea em classe executiva se o voo for muito longo?

**Resposta esperada:**

A política determina que as passagens sejam adquiridas preferencialmente na classe econômica. A aquisição de passagens em classe executiva não é permitida, salvo autorização excepcional e formal da diretoria. Portanto, a duração do voo, isoladamente, não representa uma autorização para a compra.

**Chunk esperado:** 004.

### 5. Pergunta que exige combinação de informações e cálculo

**Pergunta:**

Vou realizar uma viagem com despesas estimadas em R$ 4.000,00. Qual é o valor máximo que posso solicitar como adiantamento?

**Resposta esperada:**

O adiantamento não poderá ultrapassar 70% da estimativa total das despesas autorizadas.

R$ 4.000,00 × 70% = R$ 2.800,00.

Portanto, o valor máximo do adiantamento é de R$ 2.800,00, sujeito à aprovação do gestor imediato.

**Chunk esperado:** 012.

### 6. Pergunta que exige recuperação de informações relacionadas

**Pergunta:**

Se eu receber um adiantamento de R$ 3.000,00 e gastar apenas R$ 2.400,00, o que devo fazer?

**Resposta esperada:**

Você deverá devolver à empresa a diferença de R$ 600,00. A devolução deverá ocorrer em até 5 dias úteis após o retorno da viagem, por transferência para a conta indicada pelo setor financeiro.

**Chunk esperado:** 013.

### 7. Pergunta cuja resposta não está na base

**Pergunta:**

A empresa permite que o colaborador leve um acompanhante durante a viagem corporativa?

**Resposta esperada:**

A documentação disponibilizada não contém informações sobre a possibilidade de levar acompanhantes durante viagens corporativas. Portanto, não é possível confirmar se essa prática é permitida.

**Comportamento esperado:** o sistema não deverá inventar uma regra ou utilizar conhecimento externo para responder.

### 8. Pergunta que exige identificar uma exceção

**Pergunta:**

Preciso viajar amanhã para atender uma emergência. Posso solicitar a viagem sem cumprir os 7 dias de antecedência?

**Resposta esperada:**

Sim. Em situações excepcionais que exijam deslocamento imediato, a antecedência mínima de 7 dias poderá ser dispensada. Entretanto, a viagem deverá ser justificada formalmente e aprovada pelo gestor imediato. As demais regras de prestação de contas e comprovação de despesas continuam aplicáveis.

**Chunk esperado:** 014.

### 9. Pergunta sobre prazos de processamento

**Pergunta:**

Depois que eu enviar minha prestação de contas, quanto tempo a empresa tem para analisar e realizar o pagamento?

**Resposta esperada:**

O setor financeiro terá até 10 dias úteis, contados a partir do recebimento da prestação de contas completa, para analisar as despesas. Após a aprovação, o pagamento será realizado em até 5 dias úteis.

Assim, considerando que a prestação esteja completa e seja aprovada, os prazos máximos previstos são de até 15 dias úteis, sem considerar eventuais interrupções ou situações não previstas na política.

**Chunk esperado:** 011.

## 📋 Requisitos da solução

A solução deverá atender aos seguintes requisitos:

| ID | Requisito | Descrição |
|---|---|---|
| RF01 | 📥 Indexação | Gerar embeddings dos chunks e armazená-los em banco vetorial. |
| RF02 | 🔎 Busca semântica | Recuperar os chunks mais relevantes para a pergunta do usuário. |
| RF03 | 🔗 LangGraph | Implementar o fluxo de recuperação e geração por meio de um grafo. |
| RF04 | 🔄 Reranking | Implementar, opcionalmente, uma etapa de reordenação dos chunks. |
| RF05 | 🤖 Geração | Gerar respostas utilizando um LLM e os chunks recuperados. |
| RF06 | 📖 Fundamentação | Responder com base no conteúdo recuperado, sem inventar informações. |
| RF07 | 🚫 Ausência de contexto | Informar quando não houver informações suficientes na base documental. |

Além dos requisitos funcionais, recomenda-se que a solução:

* utilize variáveis de ambiente para armazenar credenciais e chaves de API;
* permita configurar a quantidade de chunks retornados na busca (Top-K);
* registre os chunks recuperados para facilitar a avaliação e a depuração;
* permita identificar a origem das informações utilizadas na resposta;
* separe as responsabilidades de indexação, recuperação e geração em módulos distintos;
* utilize um prompt que instrua o LLM a responder exclusivamente com base no contexto fornecido.

## 📦 Entregáveis

### Código-fonte

Um projeto organizado contendo todo o desenvolvimento da solução.

O projeto deverá incluir:

* código-fonte responsável pela indexação dos chunks no banco vetorial;
* código-fonte responsável pela implementação do grafo no LangGraph;
* pasta `chunks/` contendo os arquivos utilizados como base documental;
* arquivo de configuração com as dependências necessárias;
* instruções para instalação e execução da solução.

O código deverá estar disponível em um **repositório pessoal no GitHub**.

O **link para o repositório do GitHub deverá ser disponibilizado junto à entrega**.

### Testes

Apresente evidências dos testes realizados, contendo as perguntas utilizadas, os chunks recuperados e as respostas geradas.

As evidências deverão permitir verificar se o assistente recupera as informações relevantes e gera respostas fundamentadas na documentação.

## 🌟 Desafio adicional (opcional)

Para quem desejar aprofundar a implementação, são propostas as seguintes melhorias:

1. Implementar reranking utilizando um Cross-Encoder ou uma API especializada.
2. Implementar um nó de avaliação da relevância dos documentos recuperados.
3. Implementar um mecanismo de reformulação da pergunta quando a recuperação apresentar baixa relevância.
4. Implementar um mecanismo de avaliação da resposta gerada, verificando se todas as afirmações estão fundamentadas nos chunks recuperados.
5. Comparar diferentes modelos de embedding e avaliar seus impactos na qualidade da recuperação.

## 🎯 Objetivo final

Construir um assistente capaz de responder às dúvidas dos colaboradores sobre viagens corporativas e reembolsos de forma contextualizada, fundamentada e confiável, utilizando uma arquitetura RAG implementada com LangGraph e um banco de dados vetorial.
