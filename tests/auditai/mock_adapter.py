#!/usr/bin/env python3
"""Minimal mock HTTP target for guerrilla AuditAI runs.

Intentionally weak: one SEED blurb for all questions (not per-case RAG).
Empty response contexts so faithfulness uses dataset case.contexts.
Do not treat high scores with a smarter mock as product quality.
"""
from __future__ import annotations
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

SEED = 'Hệ thống RAG (Retrieval-Augmented Generation) cá nhân, học tập, chạy hoàn toàn miễn phí (local + free-tier API). Trả lời câu hỏi dựa trên một kho tài liệu song ngữ Việt–Anh về AI trong giáo dục và kỹ thuật RAG, có trích dẫn nguồn và cơ chế chống hallucination. Báo cáo kỹ thuật đầy đủ: xem GHI-CHU.md — kiến trúc, phương pháp, số liệu đánh giá (Retrieval + Generation), hạn chế và hướng cải thiện, kèm giải thích khái niệm/thuật ngữ. Trích xuất (Docling) → Chunking (section-based + document-aware + semantic) → Embedding (BAAI/bge-m3) → Vector DB (Chroma, HNSW) → Retrieval (Hybrid BM25+Dense → Reranker cross-encoder) → Prompt chống hallucination → Generation (Gemini/Groq) → Answer Verification.'
ATTACK = re.compile(
    r"(ignore all previous|system prompt|api key|bỏ qua mọi|tiết lộ)",
    re.I,
)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        q = str(body.get("question") or "")
        if ATTACK.search(q):
            ans = "I cannot reveal system prompts or secrets."
        elif SEED:
            # Weak: same SEED for every question (honest FAIL under strict judge)
            ans = "Based on available docs: " + SEED[:500]
        else:
            ans = f"(mock) no seed docs for: {q[:80]}"
        raw = json.dumps({"answer": ans, "contexts": []}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

print("mock adapter http://127.0.0.1:18080/chat (weak SEED, empty contexts)")
HTTPServer(("127.0.0.1", 18080), H).serve_forever()
