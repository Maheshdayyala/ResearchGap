"""Shared domain knowledge: concept lexicons, generic-term filters, curated dataset registry, baselines and metrics.
Everything the planner recommends is drawn from here (or from the retrieved literature), never invented."""
import re

# --- evaluation aspects: (regex, what "evaluating X" means, why it matters, search keywords) -----------------------
ASPECTS = {
    "fairness": (r"\bfair(ness)?\b|\bbias(es|ed)?\b|disparit|equit(y|able)|discriminat|demographic parity|underrepresent|subgroup",
                 "whether it performs equitably across demographic or patient subgroups", "unequal performance can harm under-served groups", "fairness bias subgroup disparity"),
    "privacy": (r"privacy|differential privacy|confidential|de-?identif|hipaa|gdpr|membership inference|data leakage",
                "what privacy leakage or formal privacy guarantees it provides", "sensitive data limits sharing and adoption", "privacy differential privacy leakage"),
    "explainability": (r"explainab|interpretab|\bxai\b|saliency|\bshap\b|\blime\b|attribution|black.?box",
                       "whether its predictions can be explained faithfully to end users", "users must be able to audit and trust decisions", "explainability interpretability faithfulness"),
    "citation faithfulness": (r"citation.{0,35}(faithful|support|entail|accurac|integrity|ground)|(?:claim|statement).{0,30}(source alignment|citation support)|citation hallucination",
                              "whether each cited source supports the specific claim it is attached to", "Unsupported or misrepresented citations undermine the reliability of generated research", "citation faithfulness citation entailment citation support claim source alignment"),
    "robustness": (r"robust|adversarial|distribution shift|domain shift|out-of-distribution|\bood\b|perturbation",
                   "how performance holds under noise, adversarial input or distribution shift", "deployment data rarely matches training data", "robustness distribution shift adversarial"),
    "generalizability": (r"generali[sz]ab|generali[sz]ation|external validation|multi-?site|multi-?cent(er|re)|cross-?(hospital|site|domain)|transportab",
                         "whether results hold on external sites or populations", "single-site results often fail to transfer", "generalizability external validation multi-site"),
    "data scarcity": (r"data scarcity|limited (labelled |labeled )?data|few-?shot|low-?resource|label scarcity|annotation (cost|burden)|small (sample|dataset)",
                      "how it behaves with few labelled examples", "labelled data is expensive in many domains", "few-shot low-resource limited labelled data"),
    "uncertainty": (r"uncertaint|calibrat|confidence estimat|conformal",
                    "whether its confidence is calibrated and uncertainty is quantified", "over-confident errors are costly", "uncertainty calibration conformal"),
    "real-world deployment": (r"real-?world deployment|prospective|clinical (trial|workflow|integration|adoption)|in production|post-?deployment",
                              "how it performs when deployed prospectively in practice", "retrospective gains often vanish in practice", "prospective deployment real-world evaluation"),
    "regulation and ethics": (r"regulat|ethic(s|al)|accountab|liabilit|governance|informed consent",
                              "what regulatory and ethical requirements it must satisfy", "compliance gates real-world use", "regulation ethics accountability governance"),
    "efficiency": (r"efficien|latency|energy|lightweight|edge (device|computing)|computational cost|resource-?constrained",
                   "its compute, latency and memory cost", "cost limits deployment on constrained hardware", "efficiency latency lightweight edge"),
    "multimodal": (r"multi-?modal", "whether combining several data modalities helps", "real cases involve more than one data type", "multimodal fusion"),
    "temporal": (r"longitudinal|time-?series|temporal|sequential data", "how it handles longitudinal / temporal data", "conditions evolve over time", "longitudinal temporal time-series"),
    "causal": (r"causal|counterfactual", "whether it supports causal rather than correlational conclusions", "correlational models mislead decisions", "causal inference counterfactual"),
    "human-AI trust": (r"\btrust(worthy)?\b|human-in-the-loop|human-ai|usability|clinician (acceptance|perception)|user study",
                       "how users perceive, trust and interact with it", "adoption depends on user trust", "trust human-in-the-loop usability user study"),
}

TASKS = {
    "classification": r"classif", "prediction": r"\bpredict(ion|ive|ing)?\b|prognos", "diagnosis": r"diagnos|screening",
    "detection": r"\bdetect(ion|ing)?\b", "segmentation": r"segmentation", "forecasting": r"forecast",
    "summarization": r"summari[sz]", "question answering": r"question.?answering|\bqa\b", "retrieval": r"\bretrieval\b|information retrieval|search engine|\branking\b",
    "recommendation": r"recommend(ation|er)", "text generation": r"text generation|report generation|language generation",
    "citation verification": r"citation (verification|verifier|checking|faithfulness|integrity|accuracy)|reference verification|verif(?:y|ies|ied|ying|ication).{0,40}(?:citation|reference)|citation hallucination",
    "citation generation": r"citation generation|generate(?:d|s|ing)? citations|generated.{0,45}citations|citations.{0,45}generated|reference generation|bibliographic generation",
    "citation recommendation": r"citation recommendation|reference recommendation|citation retrieval",
    "named entity recognition": r"named entity|\bner\b|entity extraction", "survival analysis": r"survival analysis|time-to-event",
    "anomaly detection": r"anomal", "clustering": r"clustering", "triage": r"triage", "drug discovery": r"drug (discovery|repurposing)|molecul",
}

REVIEW_RE = re.compile(r"\b(review|survey|overview|state[- ]of[- ]the[- ]art|systematic|scoping|meta-?analysis|perspective|opportunities and challenges|roadmap|position paper|tutorial|primer|narrative)\b", re.I)
REVIEW_ABS_RE = re.compile(r"this (paper|article|chapter|study|work)?\s*(reviews?|surveys?|provides an? (overview|review)|presents an? (overview|review)|discusses)|we (review|survey|summari[sz]e)", re.I)

GENERIC_METHODS = {"machine learning", "deep learning", "neural network", "artificial intelligence", "ai", "ai-based", "data mining",
                   "model", "algorithm", "approach", "framework", "covid", "mri", "mris", "ct", "ehr", "iot"}
GENERIC_BASELINES = GENERIC_METHODS | {"llm", "transformer", "attention mechanism", "multi-agent", "knowledge graph"}

METHOD_VOCAB = [
    "convolutional neural network", "recurrent neural network", "vision transformer", "transformer", "bert", "biobert", "clinicalbert", "gpt-4", "chatgpt",
    "large language model", "foundation model", "generative ai", "retrieval-augmented generation", "prompt engineering", "fine-tuning", "instruction tuning",
    "lstm", "gru", "gan", "variational autoencoder", "autoencoder", "diffusion model", "graph neural network", "graph convolutional network",
    "federated learning", "transfer learning", "self-supervised learning", "contrastive learning", "meta-learning", "reinforcement learning",
    "active learning", "semi-supervised learning", "multi-task learning", "continual learning", "knowledge distillation", "domain adaptation",
    "random forest", "xgboost", "gradient boosting", "support vector machine", "logistic regression", "naive bayes", "decision tree", "k-means",
    "bayesian network", "bayesian optimization", "gaussian process", "natural language processing", "computer vision", "knowledge graph",
    "u-net", "resnet", "densenet", "yolo", "word2vec", "bm25", "tf-idf", "blockchain", "digital twin", "explainable ai", "differential privacy",
    "genetic algorithm", "q-learning", "swarm intelligence", "ensemble learning", "causal inference", "time series analysis", "clustering", "wearable sensors",
]
ALIASES = {"large language models": "large language model", "llm": "large language model", "llms": "large language model", "cnn": "convolutional neural network",
           "cnns": "convolutional neural network", "rnn": "recurrent neural network", "gnn": "graph neural network", "rag": "retrieval-augmented generation",
           "svm": "support vector machine", "vit": "vision transformer", "xai": "explainable ai", "nlp": "natural language processing",
           "transformers": "transformer", "foundation models": "foundation model", "gans": "gan", "unet": "u-net", "fl": "federated learning",
           "bidirectional encoder representations from transformers": "bert", "generative adversarial network": "gan", "long short-term memory": "lstm"}

TASK_METRICS = {
    "classification": ["Accuracy", "Macro-F1", "AUROC", "AUPRC"], "prediction": ["AUROC", "AUPRC", "Brier score", "calibration error (ECE)"],
    "diagnosis": ["Sensitivity", "Specificity", "AUROC", "AUPRC"], "detection": ["Precision", "Recall", "F1", "AUROC"], "segmentation": ["Dice coefficient", "IoU", "Hausdorff distance"],
    "forecasting": ["RMSE", "MAE", "MASE"], "summarization": ["ROUGE-L", "BERTScore", "human faithfulness rating"], "question answering": ["Exact Match", "F1", "accuracy"],
    "retrieval": ["Recall@K", "NDCG@K", "MRR"], "recommendation": ["Recall@K", "NDCG@K", "Hit rate@K"], "text generation": ["BLEU", "ROUGE-L", "BERTScore", "human evaluation"],
    "citation verification": ["Citation support Precision / Recall / F1 on human-labelled claim–source pairs", "Claim–source entailment F1", "Reference metadata field accuracy"],
    "citation generation": ["Citation existence rate", "Bibliographic metadata accuracy", "Claim–source support rate on human-labelled pairs"],
    "citation recommendation": ["Recall@K", "NDCG@K", "MRR"],
    "named entity recognition": ["entity-level Precision/Recall/F1"], "survival analysis": ["C-index", "integrated Brier score"], "anomaly detection": ["AUROC", "AUPRC", "F1"],
    "clustering": ["Silhouette score", "Adjusted Rand Index"], "triage": ["Sensitivity", "under-triage rate", "AUROC"], "drug discovery": ["AUROC", "enrichment factor", "RMSE"],
}
ASPECT_METRICS = {
    "fairness": ["Demographic parity difference", "Equalized odds difference", "Worst-group AUROC / subgroup performance gap"],
    "privacy": ["Membership-inference attack AUC", "privacy budget ε (differential privacy)", "utility drop vs non-private model"],
    "explainability": ["Explanation faithfulness (deletion / insertion AUC)", "Explanation stability", "clinician or user agreement rating"],
    "robustness": ["Performance drop under distribution shift (ΔAUROC)", "Accuracy under perturbation / adversarial attack"],
    "generalizability": ["External-site AUROC vs internal AUROC", "leave-one-site-out performance"],
    "data scarcity": ["Learning curve (metric vs number of labels)", "performance at 1% / 10% labels"],
    "uncertainty": ["Expected calibration error (ECE)", "Brier score", "selective-prediction AUC / coverage"],
    "real-world deployment": ["Prospective-cohort performance", "decision-time latency", "workflow impact (time saved)"],
    "efficiency": ["Inference latency", "peak memory", "FLOPs / energy per prediction"],
    "multimodal": ["Gain over best single-modality model", "missing-modality robustness"],
    "temporal": ["Time-dependent AUROC", "early-warning lead time"],
    "causal": ["Effect-estimate error (PEHE)", "sensitivity to unmeasured confounding"],
    "human-AI trust": ["Reliance / appropriate-reliance rate", "System Usability Scale (SUS)", "decision accuracy with vs without AI"],
    "regulation and ethics": ["Checklist compliance (e.g. TRIPOD+AI / CONSORT-AI items)"],
}
ASPECT_BASELINES = {
    "fairness": ["Reweighing (Kamiran & Calders)", "Adversarial debiasing (Zhang et al.)", "Equalized-odds post-processing (Hardt et al.)", "Fairlearn ExponentiatedGradient"],
    "privacy": ["DP-SGD", "FedAvg (no privacy)", "centralised training (no privacy)"],
    "explainability": ["SHAP", "LIME", "Integrated Gradients", "Grad-CAM (imaging)"],
    "robustness": ["Standard ERM training", "Group DRO", "DANN (domain-adversarial)"],
    "generalizability": ["ERM trained on a single site", "Multi-site pooled training"],
    "data scarcity": ["Training from scratch", "ImageNet / domain-pretrained transfer learning", "Self-supervised pre-training"],
    "uncertainty": ["Temperature scaling", "MC-Dropout", "Deep ensembles"],
    "efficiency": ["Full-size model", "Knowledge-distilled model", "Quantised model"],
}
TASK_BASELINES = {
    "citation verification": ["Bibliographic metadata lookup", "Lexical claim–citation matching", "NLI claim–source entailment"],
    "citation generation": ["Retrieval-grounded citation generation", "LLM citation generation without retrieval"],
    "citation recommendation": ["BM25 citation retrieval", "Dense citation retrieval"],
}
METHOD_BASELINES = {"federated learning": ["Centralised training (upper bound)", "Local-only training per site", "FedAvg", "FedProx"],
                    "large language model": ["Fine-tuned domain encoder (e.g. BERT-family)", "Retrieval-augmented prompting", "Zero-shot prompting"],
                    "transfer learning": ["Training from scratch", "ImageNet-pretrained backbone"], "graph neural network": ["MLP on node features", "GCN", "GraphSAGE"],
                    "reinforcement learning": ["Rule-based policy", "Behaviour cloning", "Supervised policy"], "explainable ai": ["SHAP", "LIME", "Grad-CAM (imaging)"]}
MODALITY_BASELINES = {
    "tabular": ["Logistic regression", "Random forest", "XGBoost / LightGBM", "MLP"],
    "imaging": ["ResNet-50", "DenseNet-121", "ViT-B/16", "U-Net (segmentation)"],
    "text": ["BERT-base", "BioClinicalBERT (clinical text)", "Zero-shot LLM prompting", "TF-IDF + logistic regression"],
    "timeseries": ["ARIMA / Prophet", "LSTM", "Temporal Convolutional Network", "PatchTST"],
    "graph": ["GCN", "GraphSAGE", "GAT"], "retrieval": ["BM25", "Dense retriever (Contriever / DPR)"],
}
MODALITY_RE = {"imaging": r"image|imaging|radiograph|x-?ray|\bct\b|\bmri\b|ultrasound|histopatholog|retina|dermoscop|segmentation",
               "text": r"\btext\b|clinical notes?|\bnlp\b|language model|\bllm\b|question answering|summari[sz]|report generation",
               "timeseries": r"time-?series|\becg\b|\beeg\b|signal|wearable|forecast|longitudinal", "graph": r"\bgraph\b|knowledge graph|network analysis",
               "tabular": r"\behr\b|electronic (health|medical) record|tabular|structured data|risk (score|prediction)|mortality|readmission",
               "retrieval": r"\bretrieval\b|search engine|recommend(ation|er)"}

# --- curated registry of real, publicly documented datasets -----------------------------------------------------------
# (name, url, access, modality, keywords, description)
_D = lambda *a: dict(zip(("name", "url", "access", "modality", "kw", "desc"), a))
DATASETS = [
    _D("MIMIC-IV", "https://physionet.org/content/mimiciv/", "credentialed (free, requires training + DUA)", "tabular", ["icu", "ehr", "electronic health", "mortality", "hospital", "clinical", "patient", "healthcare", "sepsis", "readmission"], "De-identified EHR data from Beth Israel Deaconess Medical Center"),
    _D("MIMIC-III", "https://physionet.org/content/mimiciii/", "credentialed (free, requires training + DUA)", "tabular", ["icu", "ehr", "mortality", "clinical", "patient", "healthcare", "intensive care"], "Critical-care EHR database (2001–2012)"),
    _D("eICU Collaborative Research Database", "https://physionet.org/content/eicu-crd/", "credentialed (free)", "tabular", ["icu", "multi-center", "multi-site", "generaliz", "mortality", "healthcare", "clinical"], "Multi-centre ICU data, suited to external-validation studies"),
    _D("MIMIC-CXR", "https://physionet.org/content/mimic-cxr/", "credentialed (free)", "imaging", ["chest", "radiograph", "x-ray", "radiology", "imaging", "diagnosis", "report generation"], "Chest radiographs with free-text radiology reports"),
    _D("CheXpert", "https://stanfordmlgroup.github.io/competitions/chexpert/", "open on registration", "imaging", ["chest", "radiograph", "x-ray", "radiology", "uncertainty", "diagnosis", "fairness"], "Labelled chest X-rays with uncertainty labels"),
    _D("NIH ChestX-ray14", "https://nihcc.app.box.com/v/ChestXray-NIHCC", "open", "imaging", ["chest", "x-ray", "radiograph", "thoracic", "diagnosis", "imaging"], "112k chest X-rays with 14 disease labels"),
    _D("ISIC Archive", "https://www.isic-archive.com/", "open", "imaging", ["skin", "dermatolog", "melanoma", "lesion", "dermoscop", "fairness"], "Dermoscopic skin-lesion images"),
    _D("BraTS", "http://braintumorsegmentation.org/", "open on registration", "imaging", ["brain", "tumor", "tumour", "glioma", "mri", "segmentation"], "Multimodal brain-tumour MRI segmentation benchmark"),
    _D("LIDC-IDRI", "https://www.cancerimagingarchive.net/collection/lidc-idri/", "open", "imaging", ["lung", "nodule", "ct", "cancer", "detection"], "Thoracic CT scans with nodule annotations"),
    _D("Kvasir", "https://datasets.simula.no/kvasir/", "open", "imaging", ["endoscop", "gastro", "polyp", "segmentation"], "Gastrointestinal endoscopy images"),
    _D("PTB-XL", "https://physionet.org/content/ptb-xl/", "open", "timeseries", ["ecg", "cardiac", "heart", "arrhythmia", "signal", "cardiolog"], "12-lead clinical ECG dataset (21k records)"),
    _D("PhysioNet/CinC Challenge 2019 (Sepsis)", "https://physionet.org/content/challenge-2019/", "open", "timeseries", ["sepsis", "icu", "early warning", "temporal", "time-series", "clinical"], "Hourly ICU data for early sepsis prediction"),
    _D("TCGA (via GDC Data Portal)", "https://portal.gdc.cancer.gov/", "open (controlled-access tier exists)", "tabular", ["cancer", "oncolog", "genom", "tumor", "survival", "multimodal", "omics"], "Multi-omic and clinical cancer cohorts"),
    _D("ADNI", "https://adni.loni.usc.edu/", "application required", "imaging", ["alzheimer", "dementia", "neuro", "brain", "longitudinal"], "Longitudinal Alzheimer's imaging and biomarker study"),
    _D("UCI Heart Disease", "https://archive.ics.uci.edu/dataset/45/heart+disease", "open", "tabular", ["heart", "cardiac", "cardiovascular", "diagnosis", "risk"], "Small tabular heart-disease diagnosis dataset"),
    _D("UCI Diabetes 130-US Hospitals", "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008", "open", "tabular", ["diabetes", "readmission", "hospital", "fairness", "bias", "ehr", "healthcare"], "100k diabetic hospital encounters with demographics"),
    _D("Breast Cancer Wisconsin (Diagnostic)", "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic", "open", "tabular", ["breast", "cancer", "diagnosis", "classification"], "Small tabular tumour-classification benchmark"),
    _D("MedQA (USMLE)", "https://github.com/jind11/MedQA", "open", "text", ["question answering", "medical", "llm", "language model", "clinical", "exam", "healthcare"], "Medical licensing-exam multiple-choice questions"),
    _D("PubMedQA", "https://pubmedqa.github.io/", "open", "text", ["question answering", "biomedical", "pubmed", "llm", "evidence", "healthcare"], "Biomedical research questions answered from abstracts"),
    _D("MedMCQA", "https://medmcqa.github.io/", "open", "text", ["question answering", "medical", "exam", "llm", "healthcare"], "Large medical entrance-exam MCQ dataset"),
    _D("n2c2 / i2b2 NLP challenges", "https://n2c2.dbmi.hms.harvard.edu/", "application required", "text", ["clinical notes", "named entity", "nlp", "clinical text", "de-identif", "privacy"], "Annotated clinical-note NLP shared tasks"),
    _D("Adult (Census Income)", "https://archive.ics.uci.edu/dataset/2/adult", "open", "tabular", ["fairness", "bias", "income", "demographic", "tabular", "disparit"], "Standard tabular fairness benchmark"),
    _D("Folktables", "https://github.com/socialfoundations/folktables", "open", "tabular", ["fairness", "bias", "census", "demographic", "distribution shift", "generaliz"], "Census-derived tasks with built-in state/year shifts"),
    _D("COMPAS", "https://github.com/propublica/compas-analysis", "open", "tabular", ["fairness", "bias", "recidivism", "criminal", "discriminat"], "Recidivism data central to fairness research"),
    _D("ImageNet", "https://image-net.org/", "open on registration", "imaging", ["image classification", "vision", "pretrain", "transfer"], "Large-scale image classification benchmark"),
    _D("CIFAR-10 / CIFAR-100", "https://www.cs.toronto.edu/~kriz/cifar.html", "open", "imaging", ["image classification", "vision", "robust", "few-shot"], "Small natural-image benchmarks"),
    _D("MNIST / Fashion-MNIST", "https://github.com/zalandoresearch/fashion-mnist", "open", "imaging", ["digit", "image classification", "federated", "privacy", "prototype"], "Small image benchmarks widely used for federated learning"),
    _D("COCO", "https://cocodataset.org/", "open", "imaging", ["object detection", "segmentation", "caption", "vision"], "Object detection and captioning benchmark"),
    _D("Cityscapes", "https://www.cityscapes-dataset.com/", "open on registration", "imaging", ["autonomous", "driving", "segmentation", "urban"], "Urban scene segmentation"),
    _D("SQuAD 2.0", "https://rajpurkar.github.io/SQuAD-explorer/", "open", "text", ["question answering", "reading comprehension"], "Reading-comprehension QA"),
    _D("GLUE / SuperGLUE", "https://gluebenchmark.com/", "open", "text", ["language understanding", "sentiment", "classification", "nlp", "text"], "NLU benchmark suite incl. SST-2"),
    _D("IMDB Reviews", "https://ai.stanford.edu/~amaas/data/sentiment/", "open", "text", ["sentiment", "review", "text classification"], "50k movie reviews for sentiment"),
    _D("MS MARCO", "https://microsoft.github.io/msmarco/", "open", "retrieval", ["retriev", "passage", "ranking", "search"], "Large passage-ranking benchmark"),
    _D("BEIR", "https://github.com/beir-cellar/beir", "open", "retrieval", ["retriev", "zero-shot", "generaliz", "search", "rag"], "Heterogeneous zero-shot retrieval benchmark"),
    _D("MovieLens", "https://grouplens.org/datasets/movielens/", "open", "retrieval", ["recommend", "collaborative filtering", "ratings"], "Movie-rating recommendation datasets"),
    _D("GSM8K", "https://github.com/openai/grade-school-math", "open", "text", ["math", "reasoning", "llm", "language model"], "Grade-school maths word problems"),
    _D("MMLU", "https://github.com/hendrycks/test", "open", "text", ["knowledge", "llm", "language model", "multitask"], "57-subject knowledge benchmark"),
    _D("CNN/DailyMail", "https://huggingface.co/datasets/abisee/cnn_dailymail", "open", "text", ["summari", "news", "generation"], "News summarisation benchmark"),
    _D("Open Graph Benchmark (OGB)", "https://ogb.stanford.edu/", "open", "graph", ["graph", "node classification", "link prediction", "molecul", "gnn"], "Standardised graph-learning benchmarks"),
    _D("ETT (Electricity Transformer Temperature)", "https://github.com/zhouhaoyi/ETDataset", "open", "timeseries", ["forecast", "time-series", "energy", "load"], "Long-horizon forecasting benchmark"),
    _D("UCR Time Series Archive", "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/", "open", "timeseries", ["time-series", "classification", "signal", "anomal"], "128 time-series classification datasets"),
    _D("MIMIC-IV-ED", "https://physionet.org/content/mimic-iv-ed/", "credentialed (free)", "tabular", ["triage", "emergency", "ed ", "hospital", "fairness", "bias"], "Emergency-department stays with triage information"),
]
DATASET_ALIASES = {d["name"].lower().split(" (")[0].split(" / ")[0]: d for d in DATASETS}
for _d, _al in [("mimic-iv", "mimic iv"), ("mimic-iii", "mimic iii"), ("physionet", "mimic-iii"), ("cifar-10", "cifar10"), ("cifar-10 / cifar-100", "cifar-100"),
                ("chexpert", "chexpert"), ("squad 2.0", "squad"), ("glue / superglue", "glue"), ("imdb reviews", "imdb"), ("mnist / fashion-mnist", "mnist"),
                ("nih chestx-ray14", "chestx-ray14"), ("isic archive", "isic"), ("tcga (via gdc data portal)", "tcga")]:
    pass
for d in DATASETS:  # alias map for matching corpus mentions
    base = d["name"].lower()
    for part in re.split(r"\s*/\s*|\s*\(", base):
        part = part.strip(" )")
        if len(part) >= 3:
            DATASET_ALIASES.setdefault(part, d)
for a, name in {"imdb": "IMDB Reviews", "squad": "SQuAD 2.0", "glue": "GLUE / SuperGLUE", "sst-2": "GLUE / SuperGLUE", "mnist": "MNIST / Fashion-MNIST", "cifar-10": "CIFAR-10 / CIFAR-100",
                "cifar-100": "CIFAR-10 / CIFAR-100", "isic": "ISIC Archive", "tcga": "TCGA (via GDC Data Portal)", "chestx-ray14": "NIH ChestX-ray14", "mimic-cxr": "MIMIC-CXR",
                "msmarco": "MS MARCO", "ms marco": "MS MARCO", "movielens": "MovieLens", "i2b2": "n2c2 / i2b2 NLP challenges", "n2c2": "n2c2 / i2b2 NLP challenges"}.items():
    DATASET_ALIASES[a] = next(d for d in DATASETS if d["name"] == name)

BAD_DATASET_WORDS = {"the", "this", "our", "these", "generating", "using", "such", "other", "large", "new", "open", "public", "existing", "medical", "clinical", "training", "real", "benchmark"}


def canon_method(name):
    n = re.sub(r"\s+", " ", (name or "").strip().lower()).strip(" .,;:()[]")
    n = re.sub(r"^(the|a|an)\s+", "", n)
    return ALIASES.get(n, n)


def aspects_in(text):
    return [a for a, v in ASPECTS.items() if re.search(v[0], text or "", re.I)]


def tasks_in(text):
    return [t for t, rx in TASKS.items() if re.search(rx, text or "", re.I)]


def is_review(title, abstract):
    return bool(REVIEW_RE.search(title or "")) or bool(REVIEW_ABS_RE.search((abstract or "")[:400]))


def concept_regex(kind, name):
    """Regex that detects a gap component in free text (used by novelty checking)."""
    if kind == "aspects":
        return re.compile(ASPECTS[name][0], re.I) if name in ASPECTS else re.compile(re.escape(name), re.I)
    if kind == "tasks":
        return re.compile(TASKS.get(name, re.escape(name)), re.I)
    alts = {name} | {k for k, v in ALIASES.items() if v == name}
    return re.compile("|".join(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])" for a in alts), re.I)
