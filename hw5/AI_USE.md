# AI_USE.md — HW5 (LoRA / DialogSum)

> Rohil Utture · DATA 266 · SID4=2708
> Put this in your own words where the phrasing doesn't sound like you — the point of
> this appendix is that you can explain it.

## 1. Which parts did I use an assistant for, and which did I write myself?
I used an assistant to scaffold the notebook structure and boilerplate: the DialogSum
loading/prompt formatting, the LoRA config, the `train_lora()` training loop, and the
ROUGE / METRICS cells. I set my own parameters (SID4=2708, so SEED=2708, SLICE=708),
chose to run it locally on my RTX 3060, ran the whole notebook, read the outputs, and
wrote the Task 6 and Task 7 interpretations from my own numbers. The assistant also
flagged a bug in its own generation code (below); I verified it in my outputs, applied
the fix, and re-ran to confirm.

## 2. One specific thing it produced that was wrong. (wrong output pasted)
The `generate()` helper it wrote used beam search with no repetition control:
```python
gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams)
```
It trained fine and looked plausible, but the summaries were actually broken — they
repeated the same clause until the token budget ran out. My fine-tuned r=16 output
(Task 5) was:
```
Dawson asks #Person1# to take a dictation for her. She tells her that all office
communications are restricted to email correspondence and official memos. The use of
instant messaging programs by employees during working hours is strictly prohibited.
The use of instant messaging programs by employees during working hours is strictly
```
and the r=4 model (Task 7) degenerated into an outright loop of a wrong fact:
```
#Person1# asks #Person2# to take a dictation for Ms. Dawson. #Person2# asks #Person1#
to take a dictation for Ms. Dawson. #Person1# asks #Person1# to take
```

## 3. How did I find out? What did the failure look like?
It was visible directly in the Task 5 and Task 7 printed summaries: the same sentence
repeated two or more times and the text was cut off mid-clause at `max_new_tokens`. The
r=4 case was the clearest tell — it looped "take a dictation for Ms. Dawson" and never
produced the real content. Because the repeated n-grams are still scored against the
reference, ROUGE was being computed on degenerate text, so the metric was not trustworthy.

## 4. What did I change, and why does my version work?
I added `no_repeat_ngram_size=3` to the generation call:
```python
gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams,
                     no_repeat_ngram_size=3)
```
This forbids the decoder from emitting any 3-gram more than once, so beam search can no
longer fall into the "repeat the last clause" / loop failure mode. After re-running, my
r=16 fine-tuned summary became a single clean recap:
```
Dawson asks #Person1# to take a dictation for her. She tells her that all office
communications are restricted to email correspondence and official memos. The use of
instant messaging programs by employees during working hours is strictly prohibited.
```
The fix also exposed how misleading the old metric was: fine-tuned ROUGE-1 dropped from
0.4053 (with repetition) to 0.3764 (clean), i.e. the earlier score was partly rewarding
repeated n-grams that happened to overlap the reference. The clean numbers (baseline
ROUGE-1 0.1744 → fine-tuned 0.3764; ROUGE-L 0.1522 → 0.3059) now reflect real,
non-degenerate summaries, which is the comparison I report in Task 6.
