# Changelog

## [1.1.1] - 2025-12-07 (Hotfix)

### Added
- **OAuth 2.0 지원 추가** (GPT Desktop 호환)
  - `/.well-known/oauth-authorization-server` - OAuth 메타데이터
  - `/register` - 동적 클라이언트 등록
  - `/authorize` - 인증 엔드포인트 (자동 승인)
  - `/token` - 토큰 발급/갱신

### Fixed
- GPT Desktop "OAuth 구성 가져오기 오류" 해결
- Authorization Code + PKCE 흐름 지원

---

## [1.1.0] - 2025-12-07

### Added
- **RAG 지식 검색 기능 추가**
  - ChromaDB PersistentClient 기반 벡터 스토어
  - SentenceTransformer 'all-MiniLM-L6-v2' 임베딩
  - Leaders Decision Assistants 지식 베이스 연동

- **init_vectordb.py** - VectorDB 초기화 스크립트
  - gpt-knowledge (21개 통합 지식 파일) 로드
  - community 상세 페르소나 (1KB 이상) 로드
  - 텍스트 청킹 및 카테고리 추론
  - 총 93개 문서 인덱싱

- **PersonaVectorStore 클래스** - 지연 로딩 벡터 스토어
  - 카테고리 필터링 지원
  - 관련도 점수 반환

- **새로운 도구**
  - `search_knowledge`: 페르소나 지식 의미 검색
  - `get_knowledge_status`: RAG 상태 확인

### Changed
- requirements.txt에 RAG 의존성 추가 (chromadb, sentence-transformers)
- 서버 상태 메시지에 벡터 스토어 상태 표시

### Knowledge Sources
- **gpt-knowledge/**: 21개 통합 지식 파일
  - 01-The-Council-Executive-Advisors
  - 02-Business-Strategy-Innovation
  - 03-Startup-Entrepreneurship
  - 04-Education-Policy-International
  - 05-Analytics-Data-Science
  - 06-Backend-Development
  - 07-AI-ML-Engineering
  - 08-Frontend-Mobile-Development
  - 09-Systems-Programming
  - 10-Infrastructure-Security
  - 11-Creative-Writing
  - 12-Visual-Arts-Design
  - 13-Audio-Video-Production
  - 14-Game-Development
  - 15-Marketing-Branding
  - 16-Business-Operations
  - 17-HR-Legal-Customer
  - 18-Natural-Sciences
  - 19-Life-Sciences-Health
  - 20-Education-Teaching
  - 21-transcendent-integration-paradigms

- **community/**: 9개 상세 페르소나
  - 108-devops-engineer
  - 201-ui-ux-designer
  - 223-ux-researcher
  - 337-scrum-master
  - 405-ai-master-instructor
  - 408-management-consultant-ai
  - 410-llm-engineer
  - 411-ai-agent-developer
  - 801-world-class-leadership-coach

## [1.0.0] - Initial Release
- 페르소나 관리 MCP 서버
- 8개 기본 도구 제공
