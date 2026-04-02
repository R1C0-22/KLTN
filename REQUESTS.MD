1. Implement a preprocessing module for a temporal knowledge graph.

Requirements:
- Input: quadruples (subject, relation, object, timestamp)
- Output: natural language sentences
- Example:
  ("USA", "meet", "China", "2014-01-01")
  → "USA met China on 2014-01-01"

Tasks:
- Create functions:
  - load_dataset(path)
  - verbalize_event(s, r, o, t)
  - build_corpus(data)

Place code inside: Code/preprocessing/

=============================================================================

2. Implement entity clustering using sentence-transformers.

Requirements:
- Use sentence-transformers to embed entity names
- Use KMeans clustering
- Automatically select optimal K using silhouette score

Functions:
- embed_entities(entity_list)
- find_optimal_k(embeddings)
- run_kmeans(embeddings, k)

Output:
- cluster assignments for each entity

Place code inside: Code/clustering/

=============================================================================

3. Implement history retrieval for temporal knowledge graphs.

Tasks:
- Retrieve all events related to an entity
- Filter events with the same relation

Functions:
- get_entity_history(entity, data)
- filter_by_relation(history, relation)

Place code inside: Code/history/

=============================================================================

4. Implement short-term history selection.

Requirements:
- Sort events by timestamp
- Select last l events (default l=20)

Function:
- get_short_term(history, l)

Place code inside: Code/short_term/

=============================================================================

5. Implement long-term history filtering based on LLM scoring.

Tasks:
1. Load prompt template from prompts/filter_prompt.txt
2. Call LLM to score each event
3. Implement dynamic threshold filtering

Function:
- compute_scores_with_llm(history)
- dynamic_threshold(F, delta_t, T, alpha)
- filter_long_term(history, scores)

Follow the formula from the paper exactly.

Place code inside: Code/long_term/

=============================================================================

6. Implement analogical reasoning generation using LLM.

Tasks:
- Use prompt template from prompts/reasoning_prompt.txt
- Generate reasoning for similar events

Function:
- generate_analogical_reasoning(event, similar_events)

Place code inside: Code/analogical/

=============================================================================

7. Implement final prediction module.

Tasks:
- Combine:
  - short-term history
  - long-term filtered history
  - analogical reasoning
- Construct final prompt
- Call LLM to predict missing entity

Function:
- predict_next_object(query_event)

Place code inside: Code/inference/

=============================================================================

8. Implement a unified LLM interface.

Requirements:
- Support OpenAI and Groq
- Easy switching via config
- Handle:
  - prompt input
  - response parsing

Function:
- call_llm(prompt)

Place code inside: Code/llm/

=============================================================================

Prompt 1. filter_prompt.txt (Table 7)
Given the following historical events:

{history}

And the query event:
{query}

Evaluate the relevance of each historical event to the query.

Return a score for each event between 0 and 1.

=============================================================================

Prompt 2. reasoning_prompt.txt (Table 8)
Given the following similar historical events:

{similar_events}

Explain the reasoning process that leads to their outcomes.

Provide structured reasoning steps.

=============================================================================

Prompt 3. prediction_prompt.txt (Table 9)
Given:
Short-term history:
{short_term}

Long-term history:
{long_term}

Analogical reasoning:
{reasoning}

Predict the missing entity in:
{query}

Return only the predicted entity.
