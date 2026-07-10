"""Bước 9 (bản đầy đủ) — Generation evaluation cho TOÀN BỘ golden set (65
câu) bằng Groq (thay cho Gemini — quota free tier Gemini chỉ 20
request/ngày, liên tục hết giữa chừng trong dự án này).

Ban đầu định dùng Groq Batch API (đã xác nhận spec qua tài liệu chính thức
console.groq.com/docs/batch), nhưng khi gọi thật, Files API trả về
403 `"Not available for your plan"` — tài khoản Groq free tier này KHÔNG
được cấp quyền Batch API. Chuyển sang gọi trực tiếp endpoint đồng bộ
(/v1/chat/completions, đã test thành công 200 OK) — đơn giản hơn và không
bị chặn.

Retrieval dùng thống nhất Hybrid+Rerank (pipeline tốt nhất, giống hệt
app.py demo) cho MỌI câu hỏi — khác với lần chạy 9 câu đầu qua Gemini
trước đó (vô tình dùng dense-only mặc định), lần này đảm bảo số liệu
Generation phản ánh đúng hệ thống thật đang chạy.

Cách A (fuzzy, không cần LLM) tính lại luôn sau khi có câu trả lời, dùng
đúng hàm fuzzy_evaluate() đã có ở evaluate.py — chỉ dùng tham khảo, KHÔNG
dùng làm số liệu chính thức (đã xác nhận mất hiệu lực khi tài liệu nguồn
khác ngôn ngữ với câu trả lời, xem GHI-CHU.md).
Cách B (LLM-as-judge) dùng lại JUDGE_PROMPT đã có ở evaluate.py, đổi model
chấm điểm sang Groq/llama-3.3-70b-versatile — đây là số liệu chính thức.
"""
import json
import os
import sys
import time
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from build_prompt import build_prompt  # noqa: E402
from evaluate import JUDGE_PROMPT, fuzzy_evaluate  # noqa: E402
from retrieve import get_embed_model  # noqa: E402
from rerank import hybrid_retrieve_and_rerank  # noqa: E402

load_dotenv()

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_set.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"
RESULTS_PATH = Path(__file__).parent.parent / "data" / "generation_eval_groq.json"

START_INDEX = 0  # toan bo golden set
N_QUESTIONS = 65 - START_INDEX

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 5
DAILY_LIMIT_THRESHOLD = 120  # giay — neu retry-after vuot qua nguong nay, coi la het TPD (100.000 token/ngay,
# rieng biet voi TPM 12.000 token/phut) chu khong phai gioi han phut thoang qua -> dung lai thay vi cho giua chung


class DailyLimitExceeded(Exception):
    """Het han muc token/ngay (TPD) — khac gioi han token/phut (TPM), can
    doi sang ngay khac hoac model khac, khong nen retry ngay trong script."""


def groq_chat(prompt: str, temperature: float = 0.2) -> str:
    """Goi Groq chat completions dong bo. Uu tien doc header `retry-after`
    (Groq tra ve khi 429) vi day la con so CHINH XAC server yeu cau cho —
    phat hien duoc ca truong hop het TPM (vai giay) lan het TPD/RPD (hang
    chuc phut). Truoc day dung nham `x-ratelimit-reset-tokens` (chi phan
    anh cua so TPM, luon gan 0 ke ca khi that ra da het TPD) -> retry vo
    ich hang chuc lan lien tiep. Neu retry-after qua lon (het TPD), dung
    lai va bao loi ro rang thay vi treo script cho hang chuc phut."""
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": MODEL, "temperature": temperature, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        if resp.status_code in (429, 503):
            retry_after = resp.headers.get("retry-after")
            wait = float(retry_after) if retry_after else 10 * (attempt + 1)
            if wait > DAILY_LIMIT_THRESHOLD:
                raise DailyLimitExceeded(
                    f"Groq bao cho {wait:.0f}s ({wait/60:.1f} phut) — vuot nguong {DAILY_LIMIT_THRESHOLD}s, "
                    f"nhieu kha nang la het han muc token/ngay (TPD), khong phai token/phut (TPM). "
                    f"Body: {resp.text[:200]}"
                )
            if attempt < MAX_RETRIES - 1:
                print(f"    [{resp.status_code}] cho {wait:.1f}s (theo header retry-after)...")
                time.sleep(wait)
                continue
        resp.raise_for_status()
    raise RuntimeError("Khong the goi Groq sau nhieu lan retry")


RETRIEVAL_CACHE_PATH = Path("/tmp") / "groq_eval_retrieval_cache.json"


def main():
    if not GROQ_API_KEY:
        raise RuntimeError("Chua co GROQ_API_KEY trong .env")

    golden = json.load(open(GOLDEN_PATH, encoding="utf-8"))[START_INDEX:START_INDEX + N_QUESTIONS]

    if RETRIEVAL_CACHE_PATH.exists():
        print(f"Doc cache retrieval tu {RETRIEVAL_CACHE_PATH} (bo qua rerank CPU, ~50 phut da chay truoc do)...")
        retrieved_chunks = json.load(open(RETRIEVAL_CACHE_PATH, encoding="utf-8"))
    else:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        embed_model = get_embed_model()

        print(f"Retrieval (local, hybrid+rerank) cho {len(golden)} cau hoi...")
        retrieved_chunks = {}
        for i, g in enumerate(golden, 1):
            retrieved_chunks[g["id"]] = hybrid_retrieve_and_rerank(collection, embed_model, g["question"], k=5)
            print(f"  [{i}/{len(golden)}] {g['id']} xong retrieval")
        print("Xong retrieval.\n")
        json.dump(retrieved_chunks, open(RETRIEVAL_CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"Da luu cache retrieval vao {RETRIEVAL_CACHE_PATH} (dung lai neu can chay lai phan Groq).\n")

    # Resume: doc ket qua da co (neu ton tai) thay vi ghi de — moi lan chay
    # chi xu ly tiep cac cau CHUA xong ca 2 tieu chi (fuzzy + llm_judge),
    # de co the chay nhieu lan/nhieu ngay khi bi chan boi han muc token/ngay
    # (TPD) ma khong mat du lieu da co.
    by_id = {}
    if RESULTS_PATH.exists():
        for r in json.load(open(RESULTS_PATH, encoding="utf-8")):
            by_id[r["id"]] = r
        n_done = sum(1 for r in by_id.values() if "fuzzy" in r and "llm_judge" in r)
        print(f"Doc ket qua da co: {n_done}/{len(golden)} cau da xong ca 2 tieu chi.\n")

    def save_progress():
        ordered = [by_id.get(g["id"], {"id": g["id"], "question": g["question"]}) for g in golden]
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)

    stopped_early = False
    for g in golden:
        gid = g["id"]
        existing = by_id.get(gid, {})
        if "fuzzy" in existing and "llm_judge" in existing:
            continue  # da xong ca 2 tieu chi tu lan chay truoc, bo qua

        chunks = retrieved_chunks[gid]
        entry = {"id": gid, "question": g["question"]}
        try:
            prompt = build_prompt(g["question"], chunks)
            answer = groq_chat(prompt, temperature=0.2)
            entry["answer"] = answer
            fuzzy = fuzzy_evaluate(g["question"], answer, g["expected_answer"], chunks)
            entry["fuzzy"] = fuzzy
            print(f"[{gid}] Fuzzy -> correctness={fuzzy['correctness_score']:.2f} "
                  f"({'PASS' if fuzzy['correctness_pass'] else 'FAIL'}), "
                  f"faithfulness={fuzzy['faithfulness_score']:.2f} ({'PASS' if fuzzy['faithfulness_pass'] else 'FAIL'})")

            context_text = "\n\n".join(c["text"][:1000] for c in chunks)
            judge_prompt = JUDGE_PROMPT.format(
                question=g["question"], context=context_text,
                answer=answer, expected_answer=g["expected_answer"],
            )
            raw = groq_chat(judge_prompt, temperature=0)
            text = raw.upper()
            judge = {
                "faithfulness_pass": "FAITHFULNESS: CÓ" in text or "FAITHFULNESS:CÓ" in text,
                "correctness_pass": "CORRECTNESS: CÓ" in text or "CORRECTNESS:CÓ" in text,
                "raw": raw,
            }
            entry["llm_judge"] = judge
            print(f"       LLM-judge (Groq) -> faithfulness={'PASS' if judge['faithfulness_pass'] else 'FAIL'}, "
                  f"correctness={'PASS' if judge['correctness_pass'] else 'FAIL'}")
            by_id[gid] = entry
            save_progress()  # luu ngay sau moi cau, khong doi den cuoi — tranh mat du lieu neu bi dung giua chung
        except DailyLimitExceeded as e:
            print(f"\n[DUNG] {gid}: {e}")
            by_id[gid] = entry if "answer" in entry else existing or entry
            save_progress()
            stopped_early = True
            break
        except Exception as e:
            print(f"[{gid}] LOI: {e}")
            entry["error"] = str(e)
            by_id[gid] = entry
            save_progress()

    print(f"\nDa luu ket qua chi tiet vao {RESULTS_PATH}")
    if stopped_early:
        print("Da dung som do het han muc token/ngay (TPD) — chay lai script nay sau de tiep tuc "
              "tu cac cau chua xong (ket qua da co duoc giu nguyen, khong bi mat).")

    results = [by_id.get(g["id"], {"id": g["id"], "question": g["question"]}) for g in golden]

    ok = [r for r in results if "fuzzy" in r]
    n = len(ok)
    fuzzy_correct = sum(1 for r in ok if r["fuzzy"]["correctness_pass"])
    fuzzy_faithful = sum(1 for r in ok if r["fuzzy"]["faithfulness_pass"])
    print(f"\n=== CÁCH A (Fuzzy match, {n}/{len(golden)} câu) ===")
    print(f"Answer Correctness: {fuzzy_correct}/{n}")
    print(f"Faithfulness (proxy): {fuzzy_faithful}/{n}")

    judged = [r for r in results if "llm_judge" in r]
    m = len(judged)
    if m:
        judge_correct = sum(1 for r in judged if r["llm_judge"]["correctness_pass"])
        judge_faithful = sum(1 for r in judged if r["llm_judge"]["faithfulness_pass"])
        print(f"\n=== CÁCH B (LLM-as-judge qua Groq/{MODEL}, {m}/{len(golden)} câu) ===")
        print(f"Correctness: {judge_correct}/{m}")
        print(f"Faithfulness: {judge_faithful}/{m}")

        agree = sum(1 for r in judged if r["fuzzy"]["correctness_pass"] == r["llm_judge"]["correctness_pass"]
                    and r["fuzzy"]["faithfulness_pass"] == r["llm_judge"]["faithfulness_pass"])
        print(f"\nMức đồng thuận giữa 2 cách (cả 2 tiêu chí khớp nhau): {agree}/{m}")


if __name__ == "__main__":
    main()
