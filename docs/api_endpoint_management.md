# api.acacia.chat 엔드포인트 관리

## 현재 상태 (2026-03-03)
- **상태**: 임시 개방 (PoC 시연용)
- **목적**: `https://596428.github.io/arxiv-rag-v2/` 챗봇 외부 접근 허용
- **주의**: PoC 완료 후 반드시 닫을 것

---

## 아키텍처

```
외부 요청
  → Cloudflare (*.acacia.chat wildcard)
  → Windows Caddy (localhost:8081)
  → WSL FastAPI (localhost:8000)
  → WSL Qdrant (localhost:6333, 내부 전용)
```

---

## 관련 파일 위치

| 파일 | 경로 | 역할 |
|------|------|------|
| Cloudflare tunnel config | `C:\Users\ttyr6\.cloudflared\config.yml` | `*.acacia.chat` → Caddy 8081 |
| Caddy config | `C:\Users\ttyr6\Caddyfile` | 서브도메인별 라우팅 |
| WSL tunnel config | `/home/ajh428/.cloudflared/config.yml` | 미사용 (Windows 측이 실행) |
| 서비스 시작 bat | `/home/ajh428/projects/arxiv-rag-v1/start-acacia-services_0902.bat` | 전체 서비스 시작 |

---

## 엔드포인트 열기 (api.acacia.chat 활성화)

`C:\Users\ttyr6\Caddyfile`에 아래 블록이 있어야 함:

```caddy
# api.acacia.chat 라우팅 (arXiv RAG FastAPI)
@api host api.acacia.chat
handle @api {
    reverse_proxy localhost:8000
}
```

Caddy 재시작:
```powershell
taskkill /F /IM caddy.exe
Start-Process -FilePath "C:\Users\ttyr6\caddy\caddy.exe" -ArgumentList "run", "--config", "C:\Users\ttyr6\Caddyfile", "--adapter", "caddyfile" -WindowStyle Hidden
```

---

## 엔드포인트 닫기 (api.acacia.chat 비활성화)

`C:\Users\ttyr6\Caddyfile`에서 아래 블록 삭제 또는 주석 처리:

```caddy
# api.acacia.chat 라우팅 (arXiv RAG FastAPI)  ← 이 블록 삭제
@api host api.acacia.chat
handle @api {
    reverse_proxy localhost:8000
}
```

기본 응답 메시지도 원복:
```caddy
respond "acacia.chat - Please use a valid subdomain (code, terminal, dev, html)" 404
```

이후 Caddy 재시작.

---

## 헬스체크

```bash
# WSL 내부에서 확인
curl http://localhost:8000/api/v1/health

# 외부에서 확인 (개방 시에만 응답)
curl https://api.acacia.chat/api/v1/health
```

정상 응답:
```json
{"status": "healthy", "version": "1.0.0", "services": {"qdrant": "healthy"}}
```

---

## 보안 주의사항

- Qdrant(6333)는 절대 외부에 노출하지 말 것 (인증 없음)
- FastAPI에 rate limiting 적용되어 있음 (기본: 30req/60sec per IP)
- CORS는 `596428.github.io`, `acacia.chat` 도메인만 허용
