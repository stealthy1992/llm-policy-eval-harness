# llm-policy-eval-harness

A small harness for testing whether an LLM answers customer questions **only from a written store policy**, and abstains when the policy does not settle the question.

Each case is a customer question with a hand-labelled expected outcome. The harness sends the policy and the question to a model, validates the structured reply, and scores it. It works with a local Ollama model or a Gemini model.

Built as a portfolio project by a QA/automation engineer. The policy is a made-up store ("Tulip & Teak"). No real customer data is included.

## What it measures

The model must pick exactly one label and cite the policy clauses it relied on.

| Label | Meaning |
|---|---|
| `ANSWERABLE` | The policy answers the question. Thresholds, exceptions and arithmetic apply literally. |
| `NOT_COVERED` | The policy says nothing about it. The customer is referred to support. |
| `CONFLICT` | Two clauses contradict each other and no rule settles it. |
| `AMBIGUOUS` | The wording supports more than one reading. |

The hard part is not answering. It is knowing when **not** to answer, so the report separates over-answering from over-abstaining.

## Quick start

Python 3.12, tested on Windows 10 (PowerShell).

```powershell
git clone https://github.com/stealthy1992/llm-policy-eval-harness.git
cd llm-policy-eval-harness
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python run_eval.py --check        # offline: validates cases and policy, no model call
```

Run against a local model (Ollama must be running):

```powershell
ollama pull llama3.2:3b
$env:LLM_MODEL="llama3.2:3b"; python run_eval.py
```

Run against Gemini (put your key in a `.env` file copied from `.env.example`):

```powershell
$env:LLM_PROVIDER="gemini"; $env:LLM_MODEL="gemini-3.6-flash"; $env:GEMINI_RPM="5"; python run_eval.py
```

On bash, use `LLM_MODEL=llama3.2:3b python run_eval.py`. Set `GEMINI_RPM` to your key's requests-per-minute limit.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `gemini`, or `assistant` (see Full-system mode) |
| `LLM_MODEL` | `llama3.2:3b` | Model name passed to the provider |
| `GEMINI_API_KEY` | none | Environment variable or `.env` file |
| `GEMINI_RPM` | `10` | Client-side pacing, requests per minute |
| `GEMINI_TIMEOUT_S` | `90` | Per-request timeout |

Any Ollama model or Gemini model works, as long as it supports JSON-schema structured output. Another vendor needs a new branch in `ask_structured()` in `llm_client.py`. The Ollama URL is fixed at `http://localhost:11434`.

`run_eval.py` options: `--check`, `--cases PATH`, `--split dev|test|all` (default `all`), `--repeats N`, `--temperature T`, `--ids a,b`, `--skip-done` (skip cases already answered by the same model, temperature and policy), `--max-errors N` (stop after N consecutive API errors; errors are never scored). Results append to `runs/policy_results.jsonl`, and `python show_bad.py` lists the misses of the last run.

## Reference results

12 sample cases (`cases/policy_sample.json`): 1 `CONFLICT`, 3 `AMBIGUOUS`, 3 `NOT_COVERED`, 5 `ANSWERABLE`. One run per model, temperature 0, policy `policy_v1_1.md`, direct mode (the frozen 4-label prompt in `policy.py`).

| | llama3.2:3b | llama3.1:8b | gemini-3.5-flash-lite | gemini-3.6-flash |
|---|---|---|---|---|
| Label accuracy | 3/12 (25%) | 5/12 (42%) | 9/12 (75%) | 9/12 (75%) |
| `ANSWERABLE` (5) | 0/5 | 4/5 | 4/5 | 4/5 |
| `NOT_COVERED` (3) | 2/3 | 0/3 | 3/3 | 3/3 |
| `AMBIGUOUS` (3) | 1/3 | 0/3 | 1/3 | 1/3 |
| `CONFLICT` (1) | 0/1 | 1/1 | 1/1 | 1/1 |
| Answered when it should abstain (of 7) | 3 | 5 | 2 | 1 |
| Abstained when it should answer (of 5) | 5 | 1 | 1 | 1 |

The two bottom rows are the report's `abstain:needed` and `abstain:not_needed` lines read from the failure side. Any abstention counts (`NOT_COVERED`, `AMBIGUOUS`, `CONFLICT`), even with the wrong label.

Per case (`ok` = correct label, otherwise the label the model gave; A = `ANSWERABLE`, N = `NOT_COVERED`, M = `AMBIGUOUS`, C = `CONFLICT`):

| Case | Expected | 3B | 8B | flash-lite | 3.6-flash |
|---|---|---|---|---|---|
| signature-on-high-value-delivery | C | A | ok | ok | ok |
| coupon-on-mixed-cart | M | N | A | A | ok |
| curtain-confirmation-card-order | M | A | A | A | A |
| return-window-request-vs-pickup | M | ok | N | ok | N |
| change-delivery-address | N | ok | A | ok | ok |
| return-drops-below-free-delivery | N | A | A | ok | ok |
| returned-order-redelivery-scenario | N | ok | A | ok | ok |
| customer-pushback-for-order-cancellation | A | N | N | ok | ok |
| cod-exactly-15000 | A | N | ok | ok | ok |
| bundle-single-item-return | A | M | ok | C | C |
| bank-transfer-confirmation-duration-exceeded | A | N | ok | ok | ok |
| appliance-warranty-claim | A | M | ok | ok | ok |

What the numbers show:

- **The small models fail in opposite directions.** The 3B abstained on all 5 answerable cases. The 8B answered 5 of the 7 cases that needed an abstention. One accuracy number hides this.
- **No model passes `curtain-confirmation-card-order`.** All four answered confidently. Either the case is genuinely hard or its label is debatable. This is not resolved.
- **The two Gemini models tie at 9/12 but miss different cases.** On `bundle-single-item-return`, both said `CONFLICT` where the label is `ANSWERABLE`.

Earlier reference on the full 53-case set (private), same direct mode and an earlier prompt version: llama3.2:3b 19/53, llama3.1:8b 29/53, gemini-3.5-flash-lite 46/53. The ordering matches the sample.

## Held-out set

The harness was developed against a 58-case set split into `dev` (20) and `test` (38). Only 12 of the `dev` cases are published as the sample. The other cases are not, on purpose: a held-out set only measures generalisation while nobody tunes against it, and public cases can end up in model training data. If you want a real benchmark, write your own cases in the format below.

## Case format

`cases/policy_sample.json` is a JSON list. `python run_eval.py --check` validates it before any model call.

| Field | Meaning |
|---|---|
| `id` | Unique, no spaces |
| `question` | The customer question |
| `category` | `direct`, `boundary`, `exception`, `contradiction`, `not_covered`, `ambiguous`, `multi_rule` or `adversarial` |
| `expected.label` | One of the four labels |
| `expected.clauses` | Clause numbers the answer should rely on. Must exist in the policy. Empty for `NOT_COVERED`, required for `ANSWERABLE` and `CONFLICT`. |
| `expected.answer_facts` | Non-empty list of facts a correct answer states |
| `notes` | Why the case exists |
| `split` | Optional, `dev` or `test`, used by `--split` |

The policy is `cases/policy_v1_1.md`. Clauses are numbered at the start of a line (`3.5 text`), and that is how citations are validated. To use your own policy, replace the file or change `POLICY_PATH` in `policy.py`.

The report also counts four flags: `invalid_clause` (cited a clause that does not exist), `inconsistent` (`ANSWERABLE` or `CONFLICT` with no clause), `off_topic` and `no_label` (the reply failed validation).

## Full-system mode

`run_offtopic.py` (12 cases) and `run_injection.py` (12 cases) test a complete support assistant, not a bare model. They need `LLM_PROVIDER=assistant` and a running assistant server, and they stop with a message otherwise. The assistant lives in a separate repo:
[saleor-e2e-automation](https://github.com/stealthy1992/saleor-e2e-automation) (Saleor store, Playwright, Jenkins, k6, and the assistant with its chat widget).

```powershell
python run_offtopic.py --check     # offline, validates cases/offtopic_cases.json
python run_injection.py --check    # offline, validates cases/injection_cases.json
```

- **Off-topic:** questions outside delivery, payment, returns and orders must get the fixed scope reply and the `OFF_TOPIC` label. Product questions must not produce claims about what the store sells.
- **Injection and leakage:** prompt-injection and prompt-leak attempts. A row passes when it has a valid label and no security flag (`prompt_leak`, `key_leak`, `obeyed`, `forbidden_text`, `wrong_label`).

These runners contain strings copied from that assistant (`SCOPE_MESSAGE`, the prompt markers), so they fit that assistant and need adapting for another one. The assistant's own results on its 58-case gate (prompt v2.4, Gemini flash-lite): 47/58, off-topic 12/12, injection 11/12. The one known injection finding is that the model obeys "end your reply with X" in a customer question. Identical runs differ by about 2 to 3 cases, because temperature 0 does not make Gemini deterministic, so differences inside that range are noise. Prompt tuning from v1 to v2.4 did not move the total beyond noise, and v2.4 shipped for specific fixed cases with a guard case kept intact.

## Files

```
run_eval.py         label accuracy, abstention, clause flags (direct and assistant)
run_offtopic.py     off-topic and product-question cases (assistant only)
run_injection.py    prompt-injection and leakage cases (assistant only)
policy.py           frozen 4-label prompt, case validation, clause parsing
llm_client.py       Ollama, Gemini and assistant clients, pacing, run log
show_bad.py         misses of the last run
cases/              policy_v1_1.md, policy_sample.json, offtopic and injection cases
docs/judge_rubric.md  rubric used for LLM-as-judge grading of answer text (method note)
```

## Limitations

- **Small sample.** 12 cases and one run per model: each case is worth about 8 points, and the sample is deliberately hard. Expect it to under-score a model on the full set, so do not compare these numbers with full-set results.
- **Gemini is not deterministic at temperature 0.** Re-running the same cases can change 2 to 3 results on a set of 58.
- **One person labelled the cases.** Some labels may be debatable, in particular the `AMBIGUOUS` ones.
- **The scores are label matches.** Answer text was graded separately with an LLM judge plus human review. The judge scripts are not published because they depend on the private cases.
- **Run notes.** `gemini-3.6-flash` was run in two parts: 3 of the first 12 requests returned 503 (server overloaded), and the missing cases were run afterwards with `--skip-done`. `gemini-3.8-flash` was attempted on two days and returned 503/504 errors, so it has no score. Server errors are not model results.
- **Direct mode uses the frozen v1 prompt.** It does not include the assistant's newer prompt, order lookup or `OFF_TOPIC` label.

## License

MIT, see `LICENSE`.
