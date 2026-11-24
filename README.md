# Troubleshooting Sufficiency Classification — Raafat Saeed

Competition: https://www.kaggle.com/competitions/troubleshooting-sufficiency-classification-ml

Target: 0 sufficient; 1 insufficient. 64 training examples (42 class 0, 22 class 1); 50 unlabeled test examples. No test labels were manually assigned. IDs are identifiers only.

Evaluation: 4-fold stratified cross-validation, shuffled with seed 42. Every vocabulary, IDF and scaler is fitted inside the training fold. Cross-validation predictions are combined to calculate binary F1 for class 1. Thresholds 0.20 through 0.70, step 0.05, were selected on these predictions. These are development scores after model/threshold selection, so they are optimistic and uncertain, not independent test scores. With only 22 positive examples, score differences are fragile.

Experiments

| Run | Approach | CV F1 at 0.5 | CV F1 after threshold selection | Threshold |
|---|---|---:|---:|---:|
| run1 | Word TF-IDF + balanced logistic regression | 0.1379 | 0.5176 | 0.40 |
| run2 | Character TF-IDF + balanced logistic regression | 0.2353 | 0.5116 | 0.35 |
| run3 | Word TF-IDF + trained 16-unit neural network | 0.0800 | 0.5672 | 0.20 |
| run4 | Character TF-IDF + trained 16-unit neural network | 0.1379 | 0.4444 | 0.20 |
| run5 | Text detail statistics + balanced logistic regression | 0.4000 | 0.5294 | 0.40 |
| run6 | Text detail statistics + trained 8-unit neural network | 0.2000 | 0.5205 | 0.20 |

Our simple all-insufficient baseline has F1 0.5116. This is not the host team baseline, which has not been verified. run3 is selected with CV F1 0.5672, precision 0.4222, recall 0.8636. The gain is modest; the models are weak at this task on the available data. Default 0.5 thresholds missed most positives. Text-detail features did not outperform word TF-IDF plus MLP. No claims of high reliability are justified.

Run3: word unigrams/bigrams, sublinear TF, max 1500 TF-IDF features, trained MLP with 16 ReLU hidden units, lbfgs optimizer, alpha 1, max_iter 1500, seed 42, threshold 0.20. Refit on all 64 labeled training questions after development. best_submission.csv is identical to run3_submission.csv. Each file has 50 predictions.

Reproduce

Install dependencies from requirements.txt. Run:
`python train.py --data /path/to/troubleshooting-sufficiency-classification-ml --out results`

The supplied script produces all six candidate files and settings. No pretrained weights or API keys are required. Results.json contains model settings; cv_predictions.csv contains out-of-fold probabilities. Data_hashes.json identifies the exact inputs. Run IDs are local experiment IDs until submitted. Report only actually submitted runs on the assignment form. Public scores and host baseline scores are unknown. You still need a GitHub or Colab code link for the assignment.

Suggested Kaggle submission description:
`run3: word TF-IDF (1-2 grams, 1500 features) + trained 16-unit MLP, alpha 1, lbfgs, threshold 0.20, seed 42, 4-fold CV F1 0.5672.`

AI disclosure: ChatGPT assisted with planning, writing and executing code, evaluation and packaging. Disclose this in the required Lab 1 format; review the work and do not claim experiments you did not perform. No Kaggle submission has been made by the assistant.
