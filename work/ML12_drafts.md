# ML-12 drafts — FOR RAJKUMAR TO APPROVE (not posted, not sent)
Card asks (work/README.md): 5-min demo outline + social-post cut + 3-sentence employer summary, in the closing cells of capstone.ipynb. Numbers from work/outputs/*.json and work/FINDINGS.md.

## 5-minute demo outline
1. (0:00–0:45) Problem: an editor can't review every page; which declining pages first? Label = trend_direction "down".
2. (0:45–1:30) Data: 30,000 anonymized pages, 32 clients; 7 clients fully held out (6,163 test pages, base rate 0.511).
3. (1:30–2:15) Leakage: trend_pct gave AUC 1.000, 30-day windows 0.895 — both excluded.
4. (2:15–3:15) Tension: my hand rule was worse than random (AUC 0.463, P@50 0.36) because age works the opposite way here (decline 0.60–0.64 for young pages vs 0.42 for >333 days).
5. (3:15–4:15) Result: logistic regression AUC 0.631, P@50 0.82 vs base 0.511; row split would overstate (0.769 vs 0.626 AUC).
6. (4:15–5:00) Use + limits: decision-support review queue, not a forecast; starter sample only; paper link https://rajkumar5736602.github.io/flyrank-ml-internship/

## Social-post cut
My "old pages decay" rule ranked content-refresh candidates worse than random (AUC 0.463). On 30,000 anonymized pages with 7 clients held out, a simple logistic regression put 82% declining pages in its top 50, vs a 51% base rate. Lesson: test the intuition, and split by client — a random row split inflated AUC from 0.63 to 0.77. Built on the FlyRank ML Internship dataset (https://flyrank.ai). Paper: https://rajkumar5736602.github.io/flyrank-ml-internship/

## Employer summary (3 sentences)
I built a page-ranking model to help editors decide which content to refresh first, as part of the FlyRank ML Internship. It used 30,000 real anonymized pages from 32 clients, with 7 clients fully held out and a leakage check that removed two label-leaking features. On held-out clients, logistic regression reached Precision@50 of 0.82 against a 0.511 base rate (AUC 0.631), versus 0.36 for a hand-written rule.
