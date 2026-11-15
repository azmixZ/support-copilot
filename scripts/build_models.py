"""Build models/kb.pkl (knowledge base) and models/clf.pkl (category classifier).

The category classifier is trained on silver labels (data/labels_llm.jsonl, produced once by
scripts/label_llm.py with a stronger model). Without them it falls back to the rule-based labels.

  python scripts/build_models.py            # train on train split, report on valid, then refit on all
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

from app.knowledge import KnowledgeBase  # noqa: E402
from app.labels import has_evidence_of, priority_for  # noqa: E402
from app.pipeline import Classifier  # noqa: E402
from app.text import normalize, player_text  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"


def make_model():
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=150000)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1, token_pattern=r"[\w\[\]:.]+")),
    ])
    return Pipeline([("f", feats), ("lr", LogisticRegression(C=8, max_iter=3000, class_weight="balanced"))])


def feedback_labels():
    """(features, category) from admin corrections saved by the admin UI."""
    path = DATA / "feedback.jsonl"
    if not path.exists():
        return [], []
    xs, ys = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        t, cat = r.get("ticket"), r.get("category_corrected")
        if t and cat and cat.startswith((t.get("kind") or "") + "."):
            xs.append(featurize(t)[0])
            ys.append(cat)
    return xs, ys


def featurize(t):
    text = normalize(player_text(t["messages"]), t.get("region"))
    ev = has_evidence_of(text, t.get("images_count"))
    return Classifier.features(t["kind"], text, t.get("rule_refs_raw") or "", ev), text, ev


def train_verdict():
    """Likely admin verdict for complaints (shown to the admin as a hint)."""
    import gzip
    import numpy as np
    from app.verdict import features, training_verdict

    train =[json.loads(l) for l in gzip.open(DATA / "handout" / "tickets_train.jsonl.gz", "rt", encoding="utf-8")
             if '"complaint"' in l]
    valid = [json.loads(l) for l in open(DATA / "handout" / "tickets_valid.jsonl", encoding="utf-8")]
    xy = [(features(t), training_verdict(t)) for t in train if t["kind"] == "complaint"]
    xy = [(x, y) for x, y in xy if y]
    vxy = [(features(t), t["verdict"]) for t in valid
           if t["kind"] == "complaint" and t.get("verdict") not in (None, "other", "closed")]
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3, max_features=200000)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, token_pattern=r"[\w\[\]:.]+")),
    ])
    model = Pipeline([("f", feats), ("lr", LogisticRegression(C=4, max_iter=2000))])
    model.fit([x for x, _ in xy], [y for _, y in xy])
    proba = model.predict_proba([x for x, _ in vxy])
    pred = model.classes_[proba.argmax(1)]
    conf = proba.max(1)
    gold = np.array([y for _, y in vxy])
    rep = {"verdict_accuracy": round(float((pred == gold).mean()), 4), "n_valid": len(vxy), "n_train": len(xy),
           "majority_baseline": round(max(np.mean(gold == c) for c in set(gold)), 4)}
    for th in (0.55, 0.7):
        sel = conf >= th
        rep[f"accuracy_conf_ge_{th}"] = round(float((pred[sel] == gold[sel]).mean()), 4)
        rep[f"coverage_conf_ge_{th}"] = round(float(sel.mean()), 4)
    print("verdict model:", json.dumps(rep))
    (MODELS / "verdict_valid_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    with open(MODELS / "verdict.pkl", "wb") as f:
        pickle.dump(model, f)


def train_rulepick():
    """Main rule of a complaint, learned from the violation admins named in their answers."""
    from app.rulepick import features as rp_features, key_to_cite, training_rows
    from app.rules import RuleBase
    from app.verdict import violation_key

    rb = RuleBase(DATA / "handout" / "rules.json")
    rows = training_rows(DATA / "handout" / "tickets_train.jsonl.gz", rb)
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=150000)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[\w\[\]:.]+")),
    ])
    model = Pipeline([("f", feats), ("lr", LogisticRegression(C=4, max_iter=2000))])
    model.fit([x for x, _ in rows], [c for _, c in rows])
    valid = [json.loads(l) for l in open(DATA / "handout" / "tickets_valid.jsonl", encoding="utf-8")]
    vv = [(rp_features(t), key_to_cite(violation_key(t.get("violation"), rb.canonical), rb))
          for t in valid if t["kind"] == "complaint" and t.get("violation")]
    vv = [(x, c) for x, c in vv if c]
    pred = model.predict([x for x, _ in vv])
    rep = {"rule_accuracy_vs_admin": round(float(sum(p == c for p, (_, c) in zip(pred, vv)) / max(len(vv), 1)), 4),
           "n_valid": len(vv), "n_train": len(rows)}
    print("rule picker:", json.dumps(rep))
    (MODELS / "rulepick_valid_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    with open(MODELS / "rulepick.pkl", "wb") as f:
        pickle.dump(model, f)


def main():
    MODELS.mkdir(exist_ok=True)
    if not (MODELS / "rulepick.pkl").exists() or "--rulepick" in sys.argv:
        train_rulepick()
    if not (MODELS / "history.pkl").exists() or "--history" in sys.argv:
        from app.history import History
        print("building ticket history…")
        History.build([DATA / "handout" / "tickets_train.jsonl.gz", DATA / "handout" / "tickets_valid.jsonl"]).save(
            MODELS / "history.pkl")
    if not (MODELS / "kb.pkl").exists():
        print("building knowledge base…")
        KnowledgeBase.build(DATA / "handout" / "tickets_train.jsonl.gz").save(MODELS / "kb.pkl")
    if not (MODELS / "verdict.pkl").exists() or "--verdict" in sys.argv:
        train_verdict()

    lab_path = DATA / "labels_llm.jsonl"
    if not lab_path.exists():
        if (MODELS / "clf.pkl").exists():
            print("no silver labels here — using the classifier shipped in models/clf.pkl")
        else:
            print("no silver labels and no models/clf.pkl — the classifier falls back to keyword rules")
        return
    labels, rows = {}, []
    # Second labelling round (label_llm.py submit_more) is picked up automatically when present.
    for lab_file, sample_file in (("labels_llm.jsonl", "label_sample.jsonl"), ("labels_llm2.jsonl", "label_sample2.jsonl")):
        if not (DATA / lab_file).exists() or not (DATA / sample_file).exists():
            continue
        for l in open(DATA / lab_file, encoding="utf-8"):
            r = json.loads(l)
            labels[r["ticket_id"]] = r
        rows += [json.loads(l) for l in open(DATA / sample_file, encoding="utf-8")]
    rows = [r for r in rows if r["ticket_id"] in labels]
    # A label of the wrong kind can't be predicted anyway; drop it.
    rows = [r for r in rows if labels[r["ticket_id"]]["category"].startswith(r["kind"] + ".")]
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "valid"]
    print(f"silver labels: train={len(train)} valid={len(valid)}")

    Xtr = [featurize(r)[0] for r in train]
    ytr = [labels[r["ticket_id"]]["category"] for r in train]
    model = make_model().fit(Xtr, ytr)

    clf = Classifier(Path("/nonexistent"))
    clf.model = model
    pred_cat, pred_pri, gold_cat, gold_pri, rule_cat = [], [], [], [], []
    for r in valid:
        x, text, ev = featurize(r)
        c, _ = clf.predict(r["kind"], text, r.get("rule_refs_raw") or "", ev)
        pred_cat.append(c)
        pred_pri.append(priority_for(c, text, ev))
        gold_cat.append(labels[r["ticket_id"]]["category"])
        gold_pri.append(labels[r["ticket_id"]]["priority"])
        rule_cat.append(Classifier(Path("/nonexistent")).predict(r["kind"], text, r.get("rule_refs_raw") or "", ev)[0])
    report = {
        "category_accuracy": round(accuracy_score(gold_cat, pred_cat), 4),
        "category_accuracy_rules_only": round(accuracy_score(gold_cat, rule_cat), 4),
        "priority_macro_f1": round(f1_score(gold_pri, pred_pri, average="macro"), 4),
        "n_valid": len(valid),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(classification_report(gold_cat, pred_cat, zero_division=0))
    print(classification_report(gold_pri, pred_pri, zero_division=0))
    (MODELS / "clf_valid_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # Final model: fit on everything we have, plus admin corrections from the admin UI (data/feedback.jsonl):
    # a category an admin fixed by hand is the most reliable label we can get, so it gets a higher weight.
    fx, fy = feedback_labels()
    print(f"admin corrections used for training: {len(fx)}")
    X_all = Xtr + [featurize(r)[0] for r in valid] + fx
    y_all = ytr + gold_cat + fy
    weights = [1.0] * (len(X_all) - len(fx)) + [3.0] * len(fx)
    final = make_model()
    final.fit(X_all, y_all, lr__sample_weight=weights)
    with open(MODELS / "clf.pkl", "wb") as f:
        pickle.dump(final, f)
    print("saved models/clf.pkl")


if __name__ == "__main__":
    main()
