"""Re-measure latency / peak GPU memory for ONE model per process (no other model resident on the GPU).
Usage: python measure_isolated.py {base|gptq|teacher|student}
Same protocol as the notebook's measure_latency_memory(): batch 1, seq_len 128 random ids, 10 warmup, 50 reps, p50/p95."""
import sys, time, json, torch
device = 'cuda'
which = sys.argv[1]

def measure(model, vocab, n_warmup=10, n_reps=50, seq_len=128):
    model.eval()
    ids = torch.randint(0, vocab, (1, seq_len), device=device)
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(n_warmup): model(ids)
        torch.cuda.synchronize(); ts = []
        for _ in range(n_reps):
            torch.cuda.synchronize(); t0 = time.perf_counter(); model(ids); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts)//2], ts[int(len(ts)*0.95)], torch.cuda.max_memory_allocated(device) / 1024**3

if which == 'base':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained('artifacts/gpt2-base-fp16', dtype=torch.float16).to(device); vocab = 50257
elif which == 'gptq':
    from gptqmodel import GPTQModel, BACKEND
    q = GPTQModel.load('artifacts/gpt2-gptq-4bit', backend=BACKEND.TORCH); m = q.model if hasattr(q, 'model') else q; vocab = 50257
elif which == 'teacher':
    from transformers import BertForSequenceClassification
    m = BertForSequenceClassification.from_pretrained('teacher_model').to(device); vocab = 30522
elif which == 'student':
    from transformers import DistilBertForSequenceClassification
    m = DistilBertForSequenceClassification.from_pretrained('artifacts/best_distilled_student').to(device); vocab = 30522
p50, p95, mem = measure(m, vocab)
print(json.dumps(dict(model=which, p50_ms=round(p50,3), p95_ms=round(p95,3), peak_mem_gb=round(mem,4))))
