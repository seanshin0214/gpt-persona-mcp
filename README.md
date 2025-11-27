# GPT Persona MCP Server

ChatGPT Desktop용 페르소나 프로필 관리 MCP 서버입니다.

## 개요

Persona는 다양한 AI 페르소나 프로필을 생성, 관리, 전환할 수 있는 MCP 서버입니다. 특정 역할이나 전문성에 맞는 응답을 받을 수 있습니다.

## 기능

- **페르소나 관리**: 생성, 수정, 삭제
- **페르소나 전환**: 상황에 맞는 페르소나 적용
- **체이닝**: 여러 페르소나 순차 실행
- **커뮤니티**: 공유 페르소나 탐색 및 설치
- **분석**: 페르소나 사용 통계

## 도구 목록

| 도구 | 설명 |
|------|------|
| `create_persona` | 새 페르소나 생성 |
| `update_persona` | 페르소나 수정 |
| `delete_persona` | 페르소나 삭제 |
| `list_personas` | 페르소나 목록 조회 |
| `suggest_persona` | 상황에 맞는 페르소나 추천 |
| `chain_personas` | 페르소나 체이닝 실행 |
| `get_analytics` | 사용 통계 조회 |
| `browse_community` | 커뮤니티 페르소나 탐색 |
| `install_community_persona` | 커뮤니티 페르소나 설치 |

## 설치

```bash
pip install fastapi uvicorn aiofiles
```

## 실행

```bash
python server.py
```

서버가 `http://127.0.0.1:8767`에서 시작됩니다.

## 페르소나 예시

- **Professional**: 공식적이고 전문적인 어조
- **Casual**: 친근하고 가벼운 어조
- **Technical**: 기술적이고 상세한 설명
- **Creative**: 창의적이고 자유로운 표현

## 사용 예시

ChatGPT에서:
- "전문가 페르소나로 대답해줘"
- "새로운 페르소나 만들어줘"
- "커뮤니티 페르소나 목록 보여줘"
- "이 상황에 맞는 페르소나 추천해줘"

## 데이터 저장

페르소나 데이터는 `./personas/` 디렉토리에 JSON 형식으로 저장됩니다.

## 라이선스

MIT License
