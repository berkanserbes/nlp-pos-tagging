# Turkish Part-of-Speech Tagger

A comprehensive Part-of-Speech (POS) tagging system for Turkish text, implementing both a custom CRF-based model and Stanza integration. This project provides tools for training, evaluating, and using POS taggers with Turkish text data.

## Features

- **Custom CRF-based POS Tagger:**
  - Implements Conditional Random Fields (CRF) for sequence labeling
  - Extracts Turkish-specific morphological features
  - Supports suffix and prefix analysis
  - Includes context-aware feature extraction
  - Provides model persistence (save/load functionality)

- **Stanza Integration:**
  - Alternative implementation using Stanford's Stanza NLP library
  - Pre-trained models for Turkish language
  - Universal POS tag support

- **Interactive Interface:**
  - Command-line interface for model training and testing
  - Real-time sentence tagging
  - Detailed statistics and analysis
  - User-friendly menu system

## Project Structure

```
pos-tagging-project/
├── src/
│   ├── pos_tagger.py         # Main CRF-based POS tagger implementation
│   ├── interactive.py        # Interactive CLI interface
│   ├── stanza_pos_tagger.py  # Stanza-based implementation
│   ├── convert_to_conll.py   # Data conversion utilities
│   └── dataset.txt           # Raw dataset
└── docs/
    └── documentation.docx    # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd pos-tagging-project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run the interactive interface:

```bash
python src/interactive.py
```

The interface provides the following options:
1. Train a new model
2. Load an existing model
3. Tag sentences
4. Exit

### Training a New Model

```python
from pos_tagger import POSTagger

# Create a new tagger instance
tagger = POSTagger()

# Train the model
tagger.train('path/to/your/conll/file', test_size=0.2)

# Save the trained model
tagger.save_model('pos_model.pkl')
```

### Using a Pre-trained Model

```python
from pos_tagger import POSTagger

# Create tagger instance
tagger = POSTagger()

# Load pre-trained model
tagger.load_model('pos_model.pkl')

# Tag a sentence
sentence = "Bugün hava çok güzel."
result = tagger.predict_sentence(sentence)

# Print results
for word, tag in result:
    print(f"{word} -> {tag}")
```

### Using Stanza Implementation

```python
python src/stanza_pos_tagger.py
```

## Model Features

The CRF-based model extracts the following features for each word:

- Basic features:
  - Lowercase form
  - Case information (uppercase, title case)
  - Digit/punctuation checks
  - Length
  - Alpha/alphanumeric status

- Turkish-specific features:
  - Suffix analysis (1-3 characters)
  - Prefix analysis (1-3 characters)
  - Common Turkish morphological endings

- Contextual features:
  - Previous word features
  - Next word features
  - Sentence boundary markers

## Data Format

The project expects training data in CoNLL format:
```
word1    POS1
word2    POS2
word3    POS3

word1    POS1
word2    POS2
...
```

Each sentence is separated by an empty line.

## Performance

The model's performance is evaluated using:
- F1 Score
- Detailed classification report
- Per-tag accuracy metrics

Performance metrics are displayed during training and can be accessed through the interactive interface.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Authors

- Berkan Serbes