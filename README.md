# NLP Projects

A collection of Natural Language Processing projects exploring everything from traditional grammar rules to modern neural networks. Each project dives deep into core NLP techniques, implementing algorithms from scratch and working with real datasets to solve practical problems.

These projects span the full spectrum of NLP - from parsing sentence structure and building language models to training neural networks and engineering prompts for large language models.

---

## 📁 Projects

### 1. [`context-free-grammar-generator`](./context-free-grammar-generator)
**Context-Free Grammars & Probabilistic Sentence Generation**

This project builds context-free grammars from scratch, then adds probability to create more realistic sentence generators. 

**Key Features:**
- Hand-crafted grammar rules that capture English syntax
- Probabilistic extensions that weight common patterns more heavily
- Random sentence generation with controllable complexity
- Beautiful syntax tree visualizations

---

### 2. [`smoothed-language-modeling`](./smoothed-language-modeling)
**N-gram Models & Intelligent Text Classification**

This project tackles the classic smoothing problem in NLP, building robust bigram and trigram models that gracefully handle unseen data. We use these models to act as a spam detector that uses Bayes' theorem to classify messages.

**Key Features:**
- Sophisticated smoothing techniques (Add-λ and backoff) that prevent zero probabilities
- Text generation that captures the style of training data
- Spam detection using probabilistic reasoning
- Comprehensive evaluation with perplexity metrics

---

### 3. [`dependency-parser`](./dependency-parser)
**Syntactic Structure & Dependency Relationships**

This project transforms traditional phrase-structure trees into dependency graphs, revealing the hidden connections between words. Using linguistic rules and heuristics, it identifies which words depend on which others.

**Key Features:**
- Conversion algorithms that preserve syntactic meaning
- Rule-based headword detection using linguistic principles
- Interactive dependency graph visualization
- Analysis of complex sentence structures

---

### 4. [`hmm-pos-tagger`](./hmm-pos-tagger)
**Part-of-Speech Tagging with Hidden Markov Models**

Words wear many hats - "run" can be a verb or a noun depending on context. This project builds a part-of-speech tagger using Hidden Markov Models, learning patterns from labeled data and using the Viterbi algorithm to make smart predictions about ambiguous words.

**Key Features:**
- HMM training on real linguistic data
- Viterbi decoding for optimal sequence labeling
- Clever handling of unknown words using suffix patterns
- Detailed accuracy analysis and error investigation

---

### 5. [`rnn-language-model`](./rnn-language-model)
**Character-Level Neural Language Modeling**

Step into the world of neural networks with this from-scratch implementation of a recurrent neural network. No fancy frameworks here - just pure NumPy and mathematical intuition. Watch as the network learns to predict characters one by one, eventually generating surprisingly coherent text.

**Key Features:**
- Complete RNN implementation with forward and backward passes
- Character-level modeling that captures spelling and style patterns
- Gradient-based learning with careful numerical stability
- Text generation that showcases learned patterns

---

### 6. [`blackbox-llm-prompting`](./blackbox-llm-prompting)
**Modern Prompt Engineering & Retrieval-Augmented Generation**

Large language models are powerful, but they need the right instructions. This project explores the art and science of prompt engineering, designing few-shot examples that guide model behavior. It also implements retrieval-augmented generation, combining external knowledge with model capabilities.

**Key Features:**
- Carefully crafted prompts that elicit desired behaviors
- RAG pipeline that grounds responses in external documents
- Analysis of argument quality and reasoning patterns
- Practical techniques for working with black-box APIs

---

Each project includes detailed documentation, clean code, and thorough evaluation. Whether you're learning NLP fundamentals or exploring advanced techniques, these implementations provide both theoretical understanding and practical experience.
