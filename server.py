"""
GPT Persona MCP Server
======================
페르소나 관리 및 적용 도구 + RAG 지식 검색

원본: leaders-decision-assistants (Node.js)
GPT Desktop용 FastAPI 포팅 + ChromaDB RAG
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

import aiofiles
import aiofiles.os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# RAG Vector Store
# ============================================================================

class PersonaVectorStore:
    """페르소나 지식 RAG 벡터 스토어"""

    def __init__(self):
        self.client = None
        self.collection = None
        self.model = None
        self._initialized = False
        self.db_path = Path(__file__).parent / "data" / "chroma_db"

    def initialize(self):
        """지연 초기화"""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            if not self.db_path.exists():
                logger.warning(f"VectorDB not found at {self.db_path}")
                logger.warning("Run 'python init_vectordb.py' to initialize")
                return

            # ChromaDB 클라이언트
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )

            # 컬렉션 가져오기
            try:
                self.collection = self.client.get_collection("persona_knowledge")
                logger.info(f"Vector store loaded: {self.collection.count()} documents")
            except Exception as e:
                logger.warning(f"Collection not found: {e}")
                return

            # 임베딩 모델
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._initialized = True
            logger.info("ChromaDB RAG search enabled")

        except ImportError:
            logger.warning("chromadb or sentence-transformers not installed")
            logger.warning("Run: pip install chromadb sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")

    def search(self, query: str, category: str = None, n_results: int = 5) -> List[Dict]:
        """시맨틱 검색"""
        if not self._initialized:
            self.initialize()

        if not self._initialized or not self.collection:
            return []

        try:
            # 쿼리 임베딩
            query_embedding = self.model.encode([query])[0].tolist()

            # 필터 설정
            where_filter = None
            if category and category != "all":
                where_filter = {"category": {"$eq": category}}

            # 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # 결과 포맷
            formatted = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    relevance = max(0, 1 - distance)  # distance -> relevance

                    formatted.append({
                        "id": doc_id,
                        "content": results["documents"][0][i][:1000] + "..." if len(results["documents"][0][i]) > 1000 else results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "relevance": round(relevance, 3)
                    })

            return formatted

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_status(self) -> Dict:
        """벡터 스토어 상태"""
        if not self._initialized:
            self.initialize()

        if self._initialized and self.collection:
            return {
                "enabled": True,
                "documents": self.collection.count(),
                "path": str(self.db_path)
            }
        return {
            "enabled": False,
            "message": "Run 'python init_vectordb.py' to initialize"
        }


# 전역 벡터 스토어 인스턴스
vector_store = PersonaVectorStore()


# ============================================================================
# Persona Manager
# ============================================================================

class PersonaManager:
    """페르소나 관리자"""

    def __init__(self, persona_dir: Optional[str] = None):
        self.persona_dir = Path(persona_dir or os.path.expanduser("~/.persona"))
        self.analytics_file = self.persona_dir / ".analytics.json"
        self._ensure_dir()

    def _ensure_dir(self):
        """디렉토리 생성"""
        self.persona_dir.mkdir(parents=True, exist_ok=True)

    async def list_personas(self) -> List[str]:
        """페르소나 목록"""
        try:
            files = list(self.persona_dir.glob("*.txt"))
            return [f.stem for f in files if not f.name.startswith(".")]
        except Exception as e:
            logger.error(f"Error listing personas: {e}")
            return []

    async def get_persona(self, name: str) -> Optional[str]:
        """페르소나 읽기"""
        file_path = self.persona_dir / f"{name}.txt"
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error reading persona {name}: {e}")
            return None

    async def create_persona(self, name: str, content: str) -> bool:
        """페르소나 생성"""
        file_path = self.persona_dir / f"{name}.txt"
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.info(f"Created persona: {name}")
            return True
        except Exception as e:
            logger.error(f"Error creating persona {name}: {e}")
            return False

    async def update_persona(self, name: str, content: str) -> bool:
        """페르소나 수정"""
        file_path = self.persona_dir / f"{name}.txt"
        if not file_path.exists():
            return False
        return await self.create_persona(name, content)

    async def delete_persona(self, name: str) -> bool:
        """페르소나 삭제"""
        file_path = self.persona_dir / f"{name}.txt"
        try:
            file_path.unlink()
            logger.info(f"Deleted persona: {name}")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error deleting persona {name}: {e}")
            return False

    async def load_analytics(self) -> Dict:
        """분석 데이터 로드"""
        try:
            async with aiofiles.open(self.analytics_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except:
            return {"usage": {}, "contextPatterns": {}}

    async def save_analytics(self, data: Dict):
        """분석 데이터 저장"""
        try:
            async with aiofiles.open(self.analytics_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Error saving analytics: {e}")

    async def track_usage(self, persona_name: str, context: str = ""):
        """사용 기록 추가"""
        analytics = await self.load_analytics()

        # 사용 횟수 증가
        if persona_name not in analytics["usage"]:
            analytics["usage"][persona_name] = 0
        analytics["usage"][persona_name] += 1

        # 컨텍스트 키워드 저장
        if context:
            keywords = re.findall(r'\b\w{4,}\b', context.lower())[:5]
            if persona_name not in analytics["contextPatterns"]:
                analytics["contextPatterns"][persona_name] = {}
            for kw in keywords:
                if kw not in analytics["contextPatterns"][persona_name]:
                    analytics["contextPatterns"][persona_name][kw] = 0
                analytics["contextPatterns"][persona_name][kw] += 1

        await self.save_analytics(analytics)

    async def get_analytics(self) -> Dict:
        """분석 통계 반환"""
        analytics = await self.load_analytics()
        personas = await self.list_personas()

        return {
            "total_personas": len(personas),
            "usage_stats": analytics.get("usage", {}),
            "personas": personas
        }

    async def suggest_persona(self, context: str) -> Optional[Dict]:
        """컨텍스트 기반 페르소나 제안"""
        personas = await self.list_personas()
        if not personas:
            return None

        analytics = await self.load_analytics()
        context_lower = context.lower()

        # 탐지 규칙
        detection_rules = [
            {"keywords": ["explain", "teach", "learn", "understand", "how", "what", "why"], "persona": "teacher", "weight": 3},
            {"keywords": ["code", "function", "bug", "debug", "program", "implement", "python", "javascript"], "persona": "coder", "weight": 3},
            {"keywords": ["professional", "business", "formal", "report", "meeting"], "persona": "professional", "weight": 2},
            {"keywords": ["casual", "chat", "friendly", "hey", "talk"], "persona": "casual", "weight": 2},
            {"keywords": ["brief", "short", "quick", "summary", "concise"], "persona": "concise", "weight": 2},
            {"keywords": ["creative", "story", "imagine", "write", "novel"], "persona": "creative", "weight": 2},
        ]

        scores = {}

        # 규칙 기반 점수
        for rule in detection_rules:
            if rule["persona"] in personas:
                match_count = sum(1 for kw in rule["keywords"] if kw in context_lower)
                if match_count > 0:
                    scores[rule["persona"]] = scores.get(rule["persona"], 0) + match_count * rule["weight"]

        # 과거 패턴 기반 점수
        context_keywords = re.findall(r'\b\w{4,}\b', context_lower)
        for persona in personas:
            if persona in analytics.get("contextPatterns", {}):
                for kw in context_keywords:
                    if kw in analytics["contextPatterns"][persona]:
                        scores[persona] = scores.get(persona, 0) + 0.5

        # 최고 점수 반환
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_scores[0][1] > 1:
                return {
                    "suggested_persona": sorted_scores[0][0],
                    "confidence": min(sorted_scores[0][1] / 10, 0.95),
                    "reason": f"Context matches '{sorted_scores[0][0]}' pattern"
                }

        return None

    async def chain_personas(self, persona_names: List[str], initial_input: str) -> Dict:
        """여러 페르소나를 순차 실행"""
        results = []
        current_input = initial_input

        for name in persona_names:
            content = await self.get_persona(name)
            if content:
                results.append({
                    "persona": name,
                    "prompt": content,
                    "input": current_input
                })
                await self.track_usage(name, current_input)
            else:
                results.append({
                    "persona": name,
                    "error": f"Persona '{name}' not found"
                })

        return {
            "chain": persona_names,
            "steps": results
        }


# 전역 매니저 인스턴스
persona_manager = PersonaManager()


# ============================================================================
# MCP Server
# ============================================================================

SERVER_INFO = {
    "name": "gpt-persona-mcp",
    "version": "1.1.1",
    "description": "Persona management + RAG knowledge search for GPT Desktop (OAuth 2.0)"
}

TOOLS = [
    {
        "name": "create_persona",
        "description": "새로운 페르소나 프로필을 생성합니다",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "페르소나 이름 (예: coder, teacher, professional)"},
                "content": {"type": "string", "description": "페르소나 프롬프트 내용"}
            },
            "required": ["name", "content"]
        }
    },
    {
        "name": "update_persona",
        "description": "기존 페르소나 프로필을 수정합니다",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "수정할 페르소나 이름"},
                "content": {"type": "string", "description": "새로운 페르소나 프롬프트 내용"}
            },
            "required": ["name", "content"]
        }
    },
    {
        "name": "delete_persona",
        "description": "페르소나 프로필을 삭제합니다",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "삭제할 페르소나 이름"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "list_personas",
        "description": "사용 가능한 모든 페르소나 목록을 조회합니다",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_persona",
        "description": "특정 페르소나의 프롬프트 내용을 조회합니다",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "조회할 페르소나 이름"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "suggest_persona",
        "description": "대화 컨텍스트를 분석하여 적합한 페르소나를 제안합니다",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "분석할 대화 컨텍스트 또는 질문 내용"}
            },
            "required": ["context"]
        }
    },
    {
        "name": "chain_personas",
        "description": "여러 페르소나를 순차적으로 실행하여 단계별 처리를 수행합니다",
        "inputSchema": {
            "type": "object",
            "properties": {
                "personas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "순차 실행할 페르소나 이름 배열"
                },
                "initialInput": {"type": "string", "description": "첫 번째 페르소나에 전달할 입력"}
            },
            "required": ["personas", "initialInput"]
        }
    },
    {
        "name": "get_analytics",
        "description": "페르소나 사용 통계를 조회합니다",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "search_knowledge",
        "description": "페르소나 지식베이스를 RAG 검색합니다. 142+ 전문가 페르소나의 지식, 스킬, 전문성을 시맨틱 검색합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 질문 또는 주제"},
                "category": {
                    "type": "string",
                    "description": "카테고리 필터 (선택): Engineering, Creative, Business, AI-ML, QA, Education, Science, Leadership, Legal, all",
                    "enum": ["Engineering", "Creative", "Business", "AI-ML", "QA", "Education", "Science", "Leadership", "Legal", "all"]
                },
                "n_results": {"type": "integer", "description": "반환할 결과 수 (기본: 5)", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_knowledge_status",
        "description": "RAG 지식베이스 상태를 확인합니다",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


async def handle_tool_call(name: str, arguments: dict) -> dict:
    """도구 호출 처리"""

    if name == "create_persona":
        success = await persona_manager.create_persona(
            arguments["name"],
            arguments["content"]
        )
        msg = f"Created persona '{arguments['name']}'" if success else "Failed to create persona"
        return {"content": [{"type": "text", "text": msg}]}

    elif name == "update_persona":
        success = await persona_manager.update_persona(
            arguments["name"],
            arguments["content"]
        )
        msg = f"Updated persona '{arguments['name']}'" if success else f"Persona '{arguments['name']}' not found"
        return {"content": [{"type": "text", "text": msg}]}

    elif name == "delete_persona":
        success = await persona_manager.delete_persona(arguments["name"])
        msg = f"Deleted persona '{arguments['name']}'" if success else f"Persona '{arguments['name']}' not found"
        return {"content": [{"type": "text", "text": msg}]}

    elif name == "list_personas":
        personas = await persona_manager.list_personas()
        if personas:
            result = "## Available Personas\n\n"
            for p in sorted(personas):
                result += f"- {p}\n"
            result += f"\n**Total: {len(personas)} personas**"
        else:
            result = "No personas found. Create one with `create_persona`."
        return {"content": [{"type": "text", "text": result}]}

    elif name == "get_persona":
        content = await persona_manager.get_persona(arguments["name"])
        if content:
            await persona_manager.track_usage(arguments["name"])
            result = f"## Persona: {arguments['name']}\n\n{content}"
        else:
            result = f"Persona '{arguments['name']}' not found"
        return {"content": [{"type": "text", "text": result}]}

    elif name == "suggest_persona":
        suggestion = await persona_manager.suggest_persona(arguments["context"])
        if suggestion:
            result = json.dumps(suggestion, indent=2, ensure_ascii=False)
        else:
            result = "No suitable persona found for this context"
        return {"content": [{"type": "text", "text": result}]}

    elif name == "chain_personas":
        result = await persona_manager.chain_personas(
            arguments["personas"],
            arguments["initialInput"]
        )
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]}

    elif name == "get_analytics":
        analytics = await persona_manager.get_analytics()
        return {"content": [{"type": "text", "text": json.dumps(analytics, indent=2, ensure_ascii=False)}]}

    elif name == "search_knowledge":
        query = arguments.get("query", "")
        category = arguments.get("category", "all")
        n_results = arguments.get("n_results", 5)

        results = vector_store.search(query, category, n_results)

        if results:
            output = f"## RAG Search Results\n\n**Query**: {query}\n"
            if category and category != "all":
                output += f"**Category**: {category}\n"
            output += f"**Results**: {len(results)}\n\n---\n\n"

            for i, r in enumerate(results, 1):
                output += f"### {i}. {r['id']} (relevance: {r['relevance']})\n"
                output += f"**Source**: {r['metadata'].get('source', 'unknown')}\n"
                output += f"**Category**: {r['metadata'].get('category', 'unknown')}\n\n"
                output += f"{r['content']}\n\n---\n\n"
        else:
            status = vector_store.get_status()
            if not status.get("enabled"):
                output = "RAG search not available. Run 'python init_vectordb.py' to initialize."
            else:
                output = f"No results found for: {query}"

        return {"content": [{"type": "text", "text": output}]}

    elif name == "get_knowledge_status":
        status = vector_store.get_status()
        return {"content": [{"type": "text", "text": json.dumps(status, indent=2, ensure_ascii=False)}]}

    else:
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("GPT Persona MCP Server Starting")
    logger.info(f"Persona directory: {persona_manager.persona_dir}")
    logger.info("=" * 50)
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="GPT Persona MCP",
    description="Persona management tool for GPT Desktop",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "running", "server": SERVER_INFO}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ============================================================================
# OAuth 2.0 Endpoints for GPT Desktop
# ============================================================================

@app.get("/.well-known/oauth-authorization-server")
async def oauth_metadata(request: Request):
    """OAuth 2.0 Authorization Server Metadata"""
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse({
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "registration_endpoint": f"{base_url}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic"]
    })


@app.post("/register")
async def oauth_register(request: Request):
    """Dynamic Client Registration"""
    import secrets
    body = await request.json()
    client_id = secrets.token_urlsafe(16)
    client_secret = secrets.token_urlsafe(32)

    return JSONResponse({
        "client_id": client_id,
        "client_secret": client_secret,
        "client_name": body.get("client_name", "GPT Desktop"),
        "redirect_uris": body.get("redirect_uris", []),
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post"
    })


@app.get("/authorize")
async def oauth_authorize(
    response_type: str = "code",
    client_id: str = "",
    redirect_uri: str = "",
    state: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "S256"
):
    """Authorization Endpoint - Auto-approve and redirect"""
    import secrets
    from fastapi.responses import RedirectResponse

    # Generate authorization code
    auth_code = secrets.token_urlsafe(32)

    # Store temporarily (in production, use proper storage)
    if not hasattr(app, '_auth_codes'):
        app._auth_codes = {}
    app._auth_codes[auth_code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge
    }

    # Redirect back with code
    separator = "&" if "?" in redirect_uri else "?"
    redirect_url = f"{redirect_uri}{separator}code={auth_code}"
    if state:
        redirect_url += f"&state={state}"

    return RedirectResponse(url=redirect_url)


@app.post("/token")
async def oauth_token(request: Request):
    """Token Endpoint"""
    import secrets

    # Parse form data or JSON
    content_type = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        data = dict(form)
    else:
        data = await request.json()

    grant_type = data.get("grant_type")

    if grant_type == "authorization_code":
        # Exchange code for tokens
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        return JSONResponse({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": refresh_token
        })

    elif grant_type == "refresh_token":
        # Refresh tokens
        access_token = secrets.token_urlsafe(32)

        return JSONResponse({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600
        })

    return JSONResponse({
        "error": "unsupported_grant_type"
    }, status_code=400)


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()

        if body.get("method") == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": SERVER_INFO,
                    "capabilities": {"tools": {}}
                },
                "id": body.get("id")
            })

        elif body.get("method") == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "result": {"tools": TOOLS},
                "id": body.get("id")
            })

        elif body.get("method") == "tools/call":
            params = body.get("params", {})
            result = await handle_tool_call(params.get("name"), params.get("arguments", {}))
            return JSONResponse({
                "jsonrpc": "2.0",
                "result": result,
                "id": body.get("id")
            })

        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": body.get("id")
            })

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse({
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": str(e)},
            "id": None
        }, status_code=400)


def main():
    print("\n" + "=" * 50)
    print("  GPT Persona MCP Server")
    print("=" * 50)
    print("  URL: http://127.0.0.1:8767")
    print("  ngrok: ngrok http 8767")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8767, log_level="info")


if __name__ == "__main__":
    main()
