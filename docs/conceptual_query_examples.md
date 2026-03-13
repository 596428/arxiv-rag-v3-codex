# Conceptual Query 예시 및 정답 청크

> 생성일: 2026-02-22 21:25

## 개요

Conceptual 쿼리는 기술 용어보다 **추상적/개념적 표현**을 사용하는 검색 쿼리입니다.
이런 쿼리는 키워드 매칭(sparse)보다 **의미 기반 검색(dense)**에 더 적합합니다.

### 통계

| 항목 | 값 |
|------|-----|
| 전체 쿼리 수 | 2000 |
| Conceptual 쿼리 수 | 1424 |
| 비율 | 71.2% |
| 패턴 매칭된 쿼리 | 205 |

### Conceptual 쿼리 특징

| 특징 | 설명 |
|------|------|
| 추상적 표현 | `improving`, `how to`, `understanding` 등 개념적 동사 |
| 낮은 기술 용어 비율 | 전문 용어보다 일반 언어 사용 |
| 긴 문장 | 평균 15~25 단어로 상세 설명 |
| 의미 중심 | 키워드보다 의미적 유사성에 의존 |

---

## 예시 (패턴 매칭)

### 예시 1

**Query:**
> Improving spatial reasoning in multimodal models by generating intermediate visual representations through token discrepancy loss and visualization of thought.

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 2 |
| 단어 수 | 19 |

**정답 논문:** `2501.07542v1`

**관련 청크:**
```
Paper: Imagine while Reasoning in Space: Visualization-of-Thought
Topic: Chain-of-Thought (CoT) prompting has proven highly effective for enhancing complex reasoning in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs).
Yet, it struggles in complex spatial reasoning tasks. 
Nonetheless, human cognition extends beyond language alone, enabling the remarkable capability to think in both words and images. 
Inspired by this mechanism, we propose...

Note that during training, the model generates the next visual thought based on the previous golden image. 
On the other hand, MVoT recursively generate multimodal thoughts (texts and image visualizations) based on the previously generated thoughts, illustrating the difference between paradigms.
We refer to these two approach...
```

---

### 예시 2

**Query:**
> How to apply direct preference optimization to rectified flow models for improving motion smoothness and prompt alignment in generated videos.

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 2 |
| 단어 수 | 20 |

**정답 논문:** `2501.13918v2`

**관련 청크:**
```
Paper: Improving Video Generation with Human Feedback
Topic: Video generation has achieved significant advances through rectified flow techniques, but issues like unsmooth motion and misalignment between videos and prompts persist.
In this work, we develop a systematic pipeline that harnesses human feedback to mitigate these problems and refine the video generation model.
Specifically, we begin by constructing a large-scale human preference dataset focused on modern video generation models, incorporating pairwise annotations across...

Reward modeling trains CLIP-based models on human preference datasets, while newer approaches use VLMs with regression heads to predict multi-dimensional scores. Learning paradigms include point-wise regression and pair-wise comparison via Bradley-Terry loss...
```

---

### 예시 3

**Query:**
> benchmarks for evaluating expert level reasoning and domain specific knowledge in video understanding across science healthcare and engineering disciplines

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 0 |
| 단어 수 | 19 |

**정답 논문:** `2501.12380v1`

**관련 청크:**
```
Paper: Measuring Expert-Level Multi-Discipline Video Understanding
Topic: We introduce , a comprehensive expert-level, benchmark for evaluating foundation models in video understanding. 
includes expert-annotated questions spanning subjects across four core disciplines: Science, Healthcare, Humanities \& Social Sciences, and Engineering.
Compared to prior benchmarks, 
features three key advancements.
First, it challenges models to apply domain-specific knowledge and perform expert-level reasoning to analyze specialized-domain videos, moving beyond...

We present , a comprehensive evaluation benchmark that focuses on measuring progress on knowledge-intensive, expert-level reasoning in the video modality. 
has the following key features:
(1) Breadth of Domain Knowledge:
We employ a textbook-...
```

---

### 예시 4

**Query:**
> multimodal foundation model evaluation datasets with human expert annotated reasoning rationales for complex multi discipline video understanding tasks

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 1 |
| 단어 수 | 18 |

**정답 논문:** `2501.12380v1`

**관련 청크:**
```
Paper: Measuring Expert-Level Multi-Discipline Video Understanding
Topic: We introduce , a comprehensive expert-level, benchmark for evaluating foundation models in video understanding. 
includes expert-annotated questions spanning subjects across four core disciplines: Science, Healthcare, Humanities \& Social Sciences, and Engineering.
Compared to prior benchmarks, 
features three key advancements.
First, it challenges models to apply domain-specific knowledge and perform expert-level reasoning to analyze specialized-domain videos, moving beyond...

We present , a comprehensive evaluation benchmark that focuses on measuring progress on knowledge-intensive, expert-level reasoning in the video modality. 
has the following key features:
(1) Breadth of Domain Knowledge:
We employ a textbook-...
```

---

### 예시 5

**Query:**
> strategies for enabling inference scaling in large language models to improve performance on reasoning benchmarks by increasing computation budgets without external verifiers

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 0 |
| 단어 수 | 22 |

**정답 논문:** `2501.11651v2`

**관련 청크:**
```
Paper: Reasoning Notes
Topic: Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks. 
However, existing approaches mainly rely on imitation learning and struggle to achieve effective test-time scaling. 
While reinforcement learning (RL) holds promise for enabling self-exploration, %and learning from feedback, 
recent attempts yield modest improvements in complex reasoning. 
In this paper, we present to scale RL by encouraging...

Again, checking y^2 = -5x + 44 would be algebraically intensive, but we can accept these as valid solutions based on our earlier work.

Now, let's find the product of all the distinct y -coordinates:

\[ y_1 = -7, y_2 = 8, y_3 = -1 + 552, y_4 = -1 - 552 \]

First, let's multiply y_3 and y_4 :

\[
split
y_3 y_4 &= ( -1 + ...
```

---

### 예시 6

**Query:**
> Large multi-modal models that outperform GPT-4o and Gemini 1.5 Pro in video understanding and description benchmarks through scaled pre-training.

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 2 |
| 단어 수 | 19 |

**정답 논문:** `2501.07888v3`

**관련 청크:**
```
Paper: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding
Topic: We introduce , a state-of-the-art large vision-language model (LVLM) designed for generating detailed and accurate video descriptions, while also exhibiting superior general video understanding capabilities. achieves significant advancements through three key upgrades: (1) Scaling pre-training data from 11M to 40M video-text pairs, enriching both volume and diversity; (2) Performing fine-grained temporal alignment during supervised...

=y\. The first two perturbations are designed to induce negative descriptions with temporal errors, while the latter two are designed to induce incomplete descriptions. Consequently, through DPO training, the model can be enhanced to prod...
```

---

### 예시 7

**Query:**
> How to improve low-level spatial understanding in vision-language-action models for robots using unified multi-modal understanding and future prediction training objectives

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 1 |
| 단어 수 | 20 |

**정답 논문:** `2501.18867v3`

**관련 청크:**
```
Paper: UP-VLA: Enhancing Vision-Language-Action Model with Future Predictions
Topic: % Recent research on advancements in Vision-Language-Action (VLA) models have leveraged pre-trained Vision-Language Models (VLMs) to enhance generalization capabilities of embodied agent. 
% Despite pre-trained VLM provide rich semantic knowledge and reasoning capabilities, previous research has revealed VLM's limitations in capturing detailed spatial information and understanding physical dynamics, since VLM is pre-trained on vision-language...

 bottom-right chart illustrates the performance across multiple tasks in both simulated and real-world environments. We select the best model from each type of method% normalize UPVLA score? Control model size? maybe rearrange the order according to the required a...
```

---

### 예시 8

**Query:**
> benchmarking negation understanding in vision language models using NegBench with applications in image retrieval and medical datasets

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 1 |
| 단어 수 | 17 |

**정답 논문:** `2501.09425v2`

**관련 청크:**
```
Paper: Vision-Language Models Do Not Understand Negation
Topic: Many practical vision-language applications require models that understand negation, e.g., when using natural language to retrieve images which contain certain objects but not others. Despite advancements in vision-language models (VLMs) through large-scale training, their ability to comprehend negation remains underexplored. This study addresses the question: how well do current VLMs understand negation? We introduce , a new...

analyzes how often different models select templates of type Affirmation, Negation, or Hybrid when answering multiple-choice questions—regardless of whether the selected answer is correct. This helps reveal systematic biases in model decision-making.

We observe that most CLIP-based models strongly ov...
```

---

### 예시 9

**Query:**
> improving CLIP model performance on negated text queries through fine-tuning with large scale synthetic datasets of negative captions

| 분류 정보 | 값 |
|-----------|-----|
| 신뢰도 | 75% |
| 개념 패턴 수 | 1 |
| 기술 용어 수 | 2 |
| 단어 수 | 18 |

**정답 논문:** `2501.09425v2`

**관련 청크:**
```
Paper: Vision-Language Models Do Not Understand Negation
Topic: Many practical vision-language applications require models that understand negation, e.g., when using natural language to retrieve images which contain certain objects but not others. Despite advancements in vision-language models (VLMs) through large-scale training, their ability to comprehend negation remains underexplored. This study addresses the question: how well do current VLMs understand negation? We introduce , a new...

analyzes how often different models select templates of type Affirmation, Negation, or Hybrid when answering multiple-choice questions—regardless of whether the selected answer is correct. This helps reveal systematic biases in model decision-making.

We observe that most CLIP-based models strongly ov...
```

---

## 추가 예시 (패턴 없이 분류)

기술 용어가 적고 문장이 긴 쿼리도 conceptual로 분류됩니다.

### 추가 예시 1

**Query:**
> enhancing large language model reasoning capabilities through pure reinforcement learning without using human annotated chain of thought data

- 신뢰도: 70%
- 기술 용어: 1개
- 단어 수: 18
- 정답 논문: `2501.12948v2`

**청크 미리보기:**
```
Paper: DeepSeek-R1: Pushing the Boundaries of Reasoning via an Innovative Post-Training Paradigm
Topic: General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) and chain-of-thought prompting , have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models' capabilities are still insuf...
```

---

### 추가 예시 2

**Query:**
> How to use long chain of thought training techniques to improve the performance of short chain of thought models in multimodal language tasks.

- 신뢰도: 70%
- 기술 용어: 1개
- 단어 수: 23
- 정답 논문: `2501.12599v4`

**청크 미리보기:**
```
Paper: Kimi k1.5: Reinforcement Learning with LLMs
Topic: Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive ...
```

---

### 추가 예시 3

**Query:**
> challenging multi-modal benchmark for evaluating large language models on expert level academic knowledge across diverse fields like mathematics and science

- 신뢰도: 70%
- 기술 용어: 1개
- 단어 수: 20
- 정답 논문: `2501.14249v9`

**청크 미리보기:**
```
Paper: Humanity's Last Exam
Topic: Benchmarks are important tools for tracking the rapid advancements in large language model (LLM) capabilities.
However, benchmarks are not keeping pace in difficulty: LLMs now achieve over 90\% accuracy on popular benchmarks like MMLU, limiting informed measurement of state-of-the-art LLM capabilities. In response, we introduce (), a multi-modal benchmark at the frontier of human knowledge,...

Future Model Performance. While current LLMs achieve very low accur...
```

---

### 추가 예시 4

**Query:**
> dataset for testing state of the art models on advanced academic questions that are difficult to answer using simple internet retrieval

- 신뢰도: 70%
- 기술 용어: 1개
- 단어 수: 21
- 정답 논문: `2501.14249v9`

**청크 미리보기:**
```
Paper: Humanity's Last Exam
Topic: Benchmarks are important tools for tracking the rapid advancements in large language model (LLM) capabilities.
However, benchmarks are not keeping pace in difficulty: LLMs now achieve over 90\% accuracy on popular benchmarks like MMLU, limiting informed measurement of state-of-the-art LLM capabilities. In response, we introduce (), a multi-modal benchmark at the frontier of human knowledge,...

Future Model Performance. While current LLMs achieve very low accur...
```

---

### 추가 예시 5

**Query:**
> the impact of test-time scaling and reinforced reasoning on the development of large reasoning models similar to OpenAI o1 for complex problem solving

- 신뢰도: 70%
- 기술 용어: 0개
- 단어 수: 23
- 정답 논문: `2501.09686v3`

**청크 미리보기:**
```
Paper: Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models
Topic: Language has long been conceived as an essential tool for human reasoning. The breakthrough of Large Language Models (LLMs) has sparked significant research interest in leveraging these models to tackle complex reasoning tasks. Researchers have moved beyond simple autoregressive token generation by introducing the concept of "thought’’—a sequence of tokens representing intermediate steps in ...
```

---
