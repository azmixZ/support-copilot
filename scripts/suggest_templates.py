"""Bonus: propose new reply templates from clusters of similar tickets.

Clusters player texts (TF-IDF + MiniBatchKMeans), takes the most frequent admin replies in each
cluster and asks Claude Haiku 5.5 to turn them into one clean, rule-safe template.
Output: reports/template_suggestions.md (no raw player data, only cluster keywords and the template).

  python scripts/suggest_templates.py [n_clusters]
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from app.text import normalize, player_text  # noqa: E402

load_dotenv(ROOT / ".env")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def norm_reply(s: str) -> str:
    s = re.sub(r"[A-Z][a-z]+_[A-Z][a-z]+\d*", "<ник>", s)
    s = re.sub(r"#?<digits>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    rows = []
    with gzip.open(ROOT / "data/handout/tickets_train.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            q = normalize(player_text(t["messages"]), t.get("region"))
            a = next((m["text"] for m in t["messages"] if m["role"] == "admin"), "")
            if len(re.sub(r"\W", "", q)) >= 4 and a:
                rows.append((t["kind"], q, norm_reply(a)))
    # Cluster each kind separately; drop very common words so clusters follow the topic.
    clusters, centers = defaultdict(list), {}
    for kind in ("report", "complaint"):
        ids = [i for i, r in enumerate(rows) if r[0] == kind]
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=20, max_df=0.08, max_features=30000, sublinear_tf=True)
        X = vec.fit_transform([rows[i][1] for i in ids])
        km = MiniBatchKMeans(n_clusters=K // 2, random_state=0, batch_size=4096, n_init=3).fit(X)
        terms = np.array(vec.get_feature_names_out())
        for j, c in enumerate(km.labels_):
            clusters[(kind, c)].append(ids[j])
        for c in range(K // 2):
            centers[(kind, c)] = ", ".join(terms[np.argsort(-km.cluster_centers_[c])[:8]])

    import anthropic
    client = anthropic.Anthropic(max_retries=3)
    out = ["# Предложения новых шаблонов ответов\n",
           f"Кластеризация {len(rows)} обращений train на {K} групп. Для каждой группы: ключевые слова, доля, "
           "самые частые ответы админов (обезличены) и предложенный шаблон.\n",
           "Это шаблоны для библиотеки ответов самих администраторов, а не автоматические черновики сервиса. Фразы вроде «слежу за вами» админ может отправить только тогда, когда действительно подключился; сервис таких действий не выполняет и в своих черновиках их не обещает (там «требуется проверка администратора»).\n"]
    order = sorted(clusters, key=lambda c: -len(clusters[c]))
    for c in order:
        idx = clusters[c]
        kinds = Counter(rows[i][0] for i in idx)
        top_terms = centers[c]
        replies = Counter(rows[i][2][:300] for i in idx).most_common(6)
        prompt = ("Ниже частые ответы администраторов игры Grand Mobile на группу похожих обращений "
                  f"(ключевые слова: {top_terms}). Составь ОДИН короткий шаблон ответа на русском (1–2 предложения, в стиле этих ответов), обобщив их. "
                  "Нельзя обещать разбан, возврат, наказание другого игрока или телепорт. Номера правил упоминай только если они есть в ответах. "
                  "Верни только текст шаблона.\n\n" + "\n".join(f"- ({n} раз) {r}" for r, n in replies))
        try:
            msg = client.messages.create(model="claude-haiku-5-5", max_tokens=300, thinking={"type": "disabled"},
                                         output_config={"effort": "low"}, messages=[{"role": "user", "content": prompt}])
            tpl = next(b.text for b in msg.content if b.type == "text").strip()
        except anthropic.APIError as e:
            tpl = f"(не удалось сгенерировать: {type(e).__name__})"
        out.append(f"## {c[0]} · группа {c[1]}: {len(idx)} обращений ({len(idx) / len(rows):.1%}), "
                   f"{', '.join(f'{k} {n}' for k, n in kinds.most_common())}")
        out.append(f"**Ключевые слова:** {top_terms}\n")
        # Raw admin replies stay out of the report (data must not leave data/); only their count is shown.
        out.append(f"Обобщено из {len(replies)} самых частых ответов админов ({sum(n for _, n in replies)} случаев).")
        out.append(f"\n**Предлагаемый шаблон:** {tpl}\n")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/template_suggestions.md").write_text("\n".join(out), encoding="utf-8")
    print("written reports/template_suggestions.md")


if __name__ == "__main__":
    main()
