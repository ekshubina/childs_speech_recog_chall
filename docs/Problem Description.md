# Problem Description

In the Word Track of the On Top of Pasketti: Children's Speech Recognition Challenge, your goal is to build a model that predicts the words spoken by children in short audio clips.

Word-level automatic speech recognition (ASR) recovers the intended lexical content of speech and underpins applications such as classroom transcription, voice-driven educational tools, and accessible interfaces for young learners. Reliable word-level ASR is essential for real-world educational and accessibility use cases, yet remains particularly challenging for children's speech. This track brings together well-labeled, representative children's speech data to encourage models that generalize across ages, dialects, recording environments, and real-world classroom conditions.

## Dataset

Data for this challenge are assembled from over a dozen data sources collected under different protocols, with manual corrections and annotations added by our team. These sources have been harmonized to a common schema and released as two distinct training corpora that share the same structure but contain different data, and are hosted in separate locations for participant access.

One corpus is hosted on the DrivenData platform, while a second corpus, which follows the same schema but contains different data, is provided by TalkBank. Detailed instructions for accessing both corpora—including how to submit the required data access request form for TalkBank—are available on the [Data Download page](https://www.drivendata.org/competitions/308/childrens-word-asr/data/).

In addition, participants are welcome to incorporate external datasets for training; however, use of external data is subject to important exceptions and caveats (see [External Data and Models](https://www.drivendata.org/competitions/308/childrens-word-asr/#external-data-and-models)).

Participants are not allowed to share the competition data or use it for any purpose other than this competition. Participants cannot send the data to any third-party service or API, including but not limited to OpenAI's ChatGPT, Google's Gemini, or similar tools. For complete details, review the [competition rules](https://www.drivendata.org/competitions/308/childrens-word-asr/rules/#data-use-and-code-sharing).

### Audio

The data in this challenge are .flac audio clips of utterances from recordings of children as young as age 3 participating in a variety of controlled and semi-controlled speech tasks. These tasks include read or directed speech, prompted elicitation, and spontaneous conversational speech. The Word track and Phonetic track training data audio overlap substantially, though each track includes audio not found in the other. Training data, including audio and transcripts, can be used across tracks.

The audio clips have been scrubbed of any personally identifying information, and any adult speech that may have been present in the original recording.

The challenge dataset encompasses a broad range of U.S. regional dialects, accents, and socio-linguistic backgrounds, and includes both typically developing children and children presenting with speech sound disorders or other speech pathologies. Models are not expected to generalize to non-U.S. varieties of English. To reflect real-world deployment scenarios, a subset of the test recordings contain environmental noise characteristic of classroom and other naturalistic settings.

The DrivenData corpus is made available on the Data Download page in smaller .zip files, each containing a random subset of audio clips. The TalkBank corpus is a single zipped archive with audio files for both the Word and Phonetic tracks.

### Labels

The ground truth labels for the Word Track are normalized orthographic transcriptions of individual utterances. These transcripts represent the intended words spoken by the child rather than a verbatim transcription of the audio. As a result, disfluencies such as false starts, repetitions, and stutters are generally omitted.

Transcripts may include developmentally typical or non-standard grammatical forms common in child speech, such as using “goed” instead of “went” or “tooths” instead of “teeth.” Entire utterances in other languages have been removed; however, if a child uses a single non-English word within an otherwise English utterance, that word may be labeled in its original language (e.g., “my abuela got me those”).

Environmental noises and non-lexical sounds produced by the child (e.g., siren-like play sounds or sneezes) are not labeled. While efforts have been made to apply consistent normalization, the data were collected from multiple sources with differing annotation protocols, and some variation in labeling should be expected.

Metadata and labels for this track are provided as a UTF-8 encoded JSONL manifest. Each line corresponds to a single utterance and references exactly one associated audio file.

For each of the two corpora, the file train\_word\_transcripts.jsonl contains the following fields:

* utterance\_id (str) \- unique identifier for each utterance  
* child\_id (str) \- unique, anonymized identifier for the speaker  
* session\_id (str) \- unique identifier for the recording session; a single child\_id may be associated with multiple session\_ids  
* audio\_path (str) \- path to the corresponding .flac audio file relative to the /audio directory, following the pattern audio/{utterance\_id}.flac  
* audio\_duration\_sec (float) \- duration of the audio clip in seconds  
* age\_bucket (str) \- age range of the child at the time of recording ("3-4", "5-7", "8-11", "12+", or "unknown")  
* md5\_hash (str) \- MD5 checksum of the audio file, used for integrity verification  
* filesize\_bytes (int) \- size of the audio file in bytes  
* orthographic\_text (str) \- normalized orthographic transcription of the utterance

### Training and Test Data

The training and test splits are drawn from multiple data sources, and some sources appear exclusively in either the training or test split. A small portion of the dataset uses synthetically anonymized children’s voices. Participants are encouraged to develop models that generalize across speakers, recording conditions, and speech types.

Although all audio files have been converted to FLAC, the data vary in duration.

* Test data have been normalized to a 16 kHz sampling rate and a single channel (mono).  
* Training data have *not* been normalized. The training data vary in sampling rate and number of channels (e.g., mono vs. stereo).

## Submission Format

This is a [code execution challenge](https://drivendata.co/blog/code-execution-competitions#what-is-a-code-execution-competition)\! Rather than submitting your predicted labels, you will package your trained model and the prediction code and submit that for containerized execution.

Your code will generate a submission file that must be in JSONL format with one line per utterance. Each line must be a JSON object with the following fields:

* utterance\_id \- the unique identifier matching those provided in the submission format in the runtime environment  
* orthographic\_text \- the predicted orthographic transcript for the utterance

Information about the submission package, predict function, code and other details about the runtime environment can be found on the [Code submission format page](https://www.drivendata.org/competitions/308/childrens-word-asr/page/978/).

## Performance Metric

### Text Preprocessing

In order to ensure consistent spelling and punctuation, submissions are normalized before scoring. We apply [Whisper's English Text Normalizer](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py), which is commonly used for ASR evaluation and includes the following steps:

* Punctuation removal  
* Whitespace standardization  
* Removal of special characters and diacritics  
* Removal of content within brackets and parentheses  
* Contraction expansion (e.g., "can't" to "cannot")  
* Number and spelling standardization

We provide a [script to perform this normalization and compute the metric](https://github.com/drivendataorg/childrens-speech-recognition-runtime/blob/main/metric/score.py) in the runtime repository. Participants are encouraged to use this script to validate their submission format and reproduce the scoring behavior locally to get a sense of their performance prior to submission.

## **Metric Calculation**

Performance is evaluated using **Word Error Rate (WER)**. \[Word Error Rate (WER)\] measures the minimum number of word-level substitutions (*S*), deletions (*D*), and insertions (*I*) required to transform the predicted transcript into the reference transcript, divided by the total number of words in the reference (*N*):

\[  
WER \= \\frac{S \+ D \+ I}{N}  
\]

Since this is an error metric, a lower value is better. Words are evaluated as whole tokens (i.e., they are not split into subwords), so each word must match exactly. Partial matches (e.g., “running” vs. “run”) are counted as errors.

## Secondary Metric

A secondary metric evaluates WER only on audio recorded in natural classroom environments. These recordings include characteristics such as background noise, crosstalk, multiple speakers, varying audio quality, and other conditions typical of real-world classroom settings. Performance on Noisy Word Error Rate (Noisy WER) will be used to determine the winners of the [Noisy Classroom Bonus](https://www.drivendata.org/competitions/308/childrens-word-asr/#noisy-classroom-bonus).

### Noisy Classroom Bonus

Strong performance in noisy, real-world classroom environments is critical for educational ASR use cases. Teams that place in the top 20 on the Word Track leaderboard will be eligible for this bonus prize. The four teams with the best performance on the [Noisy Word Error Rate (Noisy WER) metric](https://www.drivendata.org/competitions/308/childrens-word-asr/page/980/#secondary-metric) will each receive $5,000.

