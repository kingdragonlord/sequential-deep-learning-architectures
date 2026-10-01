# Sequential Deep Learning: Architectural Blueprints & Data Pipelines

A modular PyTorch reference guide detailing the design, preprocessing pipelines, tensor shapes, and decoding strategies for 10 core sequence modeling problems across Natural Language Processing (NLP), Speech & Audio, Multimodal Vision-Language, and Inertial Sensor Processing.

---

## 📌 Architectural Overview & Comparison

| # | Task | Architectural Paradigm | Input Tensor $(B, T, D)$ | Target / Output Tensor | Optimization Loss | Evaluation Metric |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | **Text Generation** | Autoregressive LM | `(B, T)` | `(B, T, V)` | Cross-Entropy | Perplexity (PPL) |
| **02** | **Machine Translation** | Seq2Seq (Encoder-Decoder) | `(B, T_src)` | `(B, T_tgt, V)` | Label-Smoothed CE | BLEU, chrF |
| **03** | **Speech Recognition (ASR)** | Acoustic Seq2Seq | `(B, T_audio, 13)` | `(B, T_text, V)` | CTC / Cross-Entropy | WER, CER |
| **04** | **Named Entity Rec. (NER)** | Token Classification (Dense) | `(B, T)` | `(B, T, C)` | Masked Cross-Entropy | Entity Micro-F1 |
| **05** | **Toxic Comment Detection** | Sequence Classification | `(B, T)` | `(B, 1)` | Binary Cross-Entropy | ROC-AUC, Macro-F1 |
| **06** | **Grammar Correction (GEC)** | Monolingual Seq2Seq | `(B, T_src)` | `(B, T_tgt, V)` | Cross-Entropy | GLEU, $M^2$ Scorer |
| **07** | **Voice Cloning / TTS** | Text-to-Spectrogram + Vocoder | `(B, T_text)` | `(B, 80, T_mel)` $\rightarrow$ Audio | L1 Mel + GAN Feature Loss | MOS, PESQ, MCD |
| **08** | **Image Captioning** | Vision-Language Seq2Seq | `(B, 3, H, W)` | `(B, T_caption, V)` | Cross-Entropy | CIDEr, BLEU-4 |
| **09** | **Human Activity Rec. (HAR)** | Temporal Feature Classification | `(B, T, D_sensor)` | `(B, C)` | Cross-Entropy | Weighted F1 |
| **10** | **Speech Emotion Rec. (SER)** | Acoustic Sequence Classification | `(B, T, D_mfcc)` | `(B, C)` | Cross-Entropy | UAR, Balanced Acc |

---

## 🛠️ Task Specifications

### 1. Autoregressive Text Generation
* **Paradigm:** Causal Language Modeling.
* **Architecture:** Token Embedding $\rightarrow$ Stacked Unidirectional LSTM / Causal Transformer Blocks $\rightarrow$ Linear Projection $\rightarrow$ Softmax.
* **Pipeline:** Sliding window over tokenized corpora using a fixed `max_length` and `stride`, shifting target labels $Y$ forward by one index ($x_t \rightarrow x_{t+1}$).
* **Tensor Flow:** `(B, T)` $\rightarrow$ `(B, T, D_model)` $\rightarrow$ `(B, T, V)`.
* **Inference Strategy:** Temperature-scaled sampling, Top-$k$, or Top-$p$ (Nucleus) decoding.

### 2. Machine Translation
* **Paradigm:** Sequence-to-Sequence (Encoder-Decoder).
* **Architecture:** Subword Embedding (BPE/SentencePiece) $\rightarrow$ Transformer Encoder $\rightarrow$ Cross-Attention Decoder $\rightarrow$ Vocabulary Projection.
* **Pipeline:** Source/target paired corpus preprocessed using subword segmentation to gracefully handle compound nouns and continuous scripts; batch-padded using `<pad>` masks with `<bos>` and `<eos>` sequence boundaries.
* **Tensor Flow:** Encoder: `(B, T_src)` $\rightarrow$ `(B, T_src, D)`; Decoder: `(B, T_tgt)` $\rightarrow$ `(B, T_tgt, V)`.
* **Inference Strategy:** Beam Search decoding with length penalty.

### 3. Automatic Speech Recognition (ASR)
* **Paradigm:** Audio Sequence-to-Sequence.
* **Architecture:** Audio Preprocessor (13-dim MFCC / 80-dim Filterbanks) $\rightarrow$ Conformer / Transformer Encoder $\rightarrow$ CTC Decoder or Autoregressive Text Decoder.
* **Pipeline:** Raw mono audio resampled to 16 kHz, converted into spectral frames across a rolling window ($N_{\text{fft}} = 400$, $\text{hop} = 160$), and padded to maximum temporal duration via `pad_sequence`.
* **Tensor Flow:** `(B, T_audio, 13)` $\rightarrow$ `(B, T_audio, D)` $\rightarrow$ `(B, T_text, V)`.

### 4. Named Entity Recognition (NER)
* **Paradigm:** Dense Token Classification / Sequence Labeling.
* **Architecture:** Token Embedding $\rightarrow$ Bidirectional Encoder (BiLSTM / BERT-style Transformer) $\rightarrow$ Token-Level Linear Head (or Linear-Chain CRF).
* **Pipeline:** Subword/word tokenization aligned with BIO/BILOU entity spans. Loss computed strictly over non-padded tokens using an ignore index ($C_{\text{pad}} = -100$).
* **Tensor Flow:** `(B, T)` $\rightarrow$ `(B, T, D)` $\rightarrow$ `(B, T, C_{\text{labels}})`.

### 5. Toxic Comment Classification
* **Paradigm:** Sequence Classification.
* **Architecture:** Subword Embedding $\rightarrow$ Transformer Encoder $\rightarrow$ Mean/Attention Pooling Layer $\rightarrow$ Dense Classification Head $\rightarrow$ Sigmoid.
* **Pipeline:** Cleaned comments tokenized and padded; binary multi-label or single-label target representation.
* **Tensor Flow:** `(B, T)` $\rightarrow$ `(B, T, D)` $\rightarrow$ Pooling: `(B, D)` $\rightarrow$ `(B, 1)`.

### 6. Grammar Error Correction (GEC)
* **Paradigm:** Monolingual Sequence-to-Sequence Rewriting.
* **Architecture:** Transformer Encoder-Decoder with tied input-output embeddings and copy-mechanism / pointer networks.
* **Pipeline:** Synthetically corrupted or human-annotated ungrammatical/grammatical pairs tokenized with shared subword vocabularies.
* **Tensor Flow:** `(B, T_src)` $\rightarrow$ `(B, T_tgt, V)`.

### 7. Voice Cloning & Speech Synthesis (TTS)
* **Paradigm:** Two-Stage Acoustic Synthesis (Text-to-Spectrogram + Neural Vocoder).
* **Architecture:** Character/Phoneme Encoder $\rightarrow$ Duration Predictor & Variance Adaptor $\rightarrow$ Mel-Spectrogram Decoder $\rightarrow$ HiFi-GAN / WaveNet Vocoder.
* **Pipeline:** Aligned text transcripts and normalized mel-spectrogram matrices with speaker embeddings for voice personalization.
* **Tensor Flow:** `(B, T_text)` $\rightarrow$ `(B, 80, T_mel)` $\rightarrow$ `(B, 1, T_samples)`.

### 8. Image Captioning
* **Paradigm:** Vision-Language Encoder-Decoder.
* **Architecture:** Pretrained Vision Backbone (ResNet / ViT) $\rightarrow$ Linear Feature Projection $\rightarrow$ Cross-Attention Autoregressive Transformer Decoder.
* **Pipeline:** Resized and normalized image tensors $(3 \times H \times W)$ paired with tokenized textual captions bounded by `<bos>` and `<eos>`.
* **Tensor Flow:** Vision Input: `(B, 3, H, W)` $\rightarrow$ Visual Tokens: `(B, N_patches, D)` $\rightarrow$ Caption Logits: `(B, T_seq, V)`.

### 9. Human Activity Recognition (HAR)
* **Paradigm:** Multi-Channel Sensor Sequence Classification.
* **Architecture:** Multi-Axis IMU Signals (Accelerometer + Gyroscope) $\rightarrow$ 1D-CNN Temporal Feature Extractor $\rightarrow$ Bidirectional GRU / Transformer $\rightarrow$ Global Pooling $\rightarrow$ Softmax.
* **Pipeline:** Sliding window segmentation over continuous sensor channels with fixed overlapping steps; batch-stacked along channel dimensions.
* **Tensor Flow:** `(B, T, D_channels)` $\rightarrow$ `(B, T', D_hidden)` $\rightarrow$ `(B, C_{\text{activities}})`.

### 10. Speech Emotion Recognition (SER)
* **Paradigm:** Audio Sequence Classification.
* **Architecture:** Log-Mel Spectrogram / MFCC $\rightarrow$ 2D-CNN Spatial Filterbank $\rightarrow$ Temporal Attention Recurrent Unit $\rightarrow$ Dense Classification Head.
* **Pipeline:** Standardized audio segments transformed to time-frequency representations, zero-padded, and mapped to categorical emotional states (e.g., Happy, Angry, Neutral).
* **Tensor Flow:** `(B, T, D_features)` $\rightarrow$ Temporal Summarization: `(B, D)` $\rightarrow$ `(B, C_{\text{emotions}})`.

---

## 💻 Tech Stack & Requirements

* **Framework:** PyTorch (`torch`, `torch.nn`, `torch.nn.utils.rnn`)
* **Audio Processing:** `torchaudio`
* **Data Handling:** `collections.Counter`, `urllib`
