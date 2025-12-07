#!/usr/bin/env python3
"""
Persona Knowledge Base VectorDB Initializer
============================================
Leaders Decision Assistants 지식을 ChromaDB에 임베딩
"""

import os
import sys
from pathlib import Path

# ChromaDB 및 임베딩 라이브러리
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Required packages not installed. Run:")
    print("pip install chromadb sentence-transformers")
    sys.exit(1)


# ============================================================================
# 경로 설정
# ============================================================================

# Leaders Decision Assistants 지식 베이스 경로
LEADERS_KNOWLEDGE_DIR = Path(os.path.expanduser("~")) / "Documents" / "leaders-decision-assistants" / "gpt-knowledge"
LEADERS_COMMUNITY_DIR = Path(os.path.expanduser("~")) / "Documents" / "leaders-decision-assistants" / "community"

# VectorDB 저장 경로
VECTOR_DB_PATH = Path(__file__).parent / "data" / "chroma_db"


# ============================================================================
# 지식 로더
# ============================================================================

def load_gpt_knowledge() -> list:
    """gpt-knowledge 폴더에서 통합 지식 파일 로드"""
    documents = []

    if not LEADERS_KNOWLEDGE_DIR.exists():
        print(f"Warning: Knowledge directory not found: {LEADERS_KNOWLEDGE_DIR}")
        return documents

    for file_path in sorted(LEADERS_KNOWLEDGE_DIR.glob("*.txt")):
        try:
            content = file_path.read_text(encoding='utf-8')

            # 파일명에서 카테고리 추출
            filename = file_path.stem
            parts = filename.split('-', 1)
            category = parts[1] if len(parts) > 1 else filename

            # 문서가 너무 크면 청크로 분할
            chunks = split_into_chunks(content, max_chars=4000)

            for i, chunk in enumerate(chunks):
                doc_id = f"knowledge-{filename}-{i}" if len(chunks) > 1 else f"knowledge-{filename}"
                documents.append({
                    "id": doc_id,
                    "content": chunk,
                    "metadata": {
                        "source": "gpt-knowledge",
                        "category": category,
                        "file": filename
                    }
                })

            print(f"  Loaded: {filename} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")

    return documents


def load_community_personas() -> list:
    """community 폴더에서 상세 페르소나 파일 로드 (큰 파일만)"""
    documents = []

    if not LEADERS_COMMUNITY_DIR.exists():
        print(f"Warning: Community directory not found: {LEADERS_COMMUNITY_DIR}")
        return documents

    # 1KB 이상인 상세 페르소나만 로드
    for file_path in sorted(LEADERS_COMMUNITY_DIR.glob("*.txt")):
        try:
            file_size = file_path.stat().st_size
            if file_size < 1000:  # 1KB 미만은 메타데이터만이므로 스킵
                continue

            content = file_path.read_text(encoding='utf-8')
            filename = file_path.stem

            # 번호와 이름 분리
            parts = filename.split('-', 1)
            persona_id = parts[0] if parts[0].isdigit() else "000"
            persona_name = parts[1] if len(parts) > 1 else filename

            # 카테고리 추론
            category = infer_category(persona_id)

            # 청크 분할
            chunks = split_into_chunks(content, max_chars=3000)

            for i, chunk in enumerate(chunks):
                doc_id = f"persona-{filename}-{i}" if len(chunks) > 1 else f"persona-{filename}"
                documents.append({
                    "id": doc_id,
                    "content": chunk,
                    "metadata": {
                        "source": "community",
                        "category": category,
                        "persona_id": persona_id,
                        "persona_name": persona_name
                    }
                })

            print(f"  Loaded: {filename} ({file_size}B, {len(chunks)} chunks)")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")

    return documents


def infer_category(persona_id: str) -> str:
    """페르소나 ID로 카테고리 추론"""
    try:
        num = int(persona_id)
        if 100 <= num < 200:
            return "Engineering"
        elif 200 <= num < 300:
            return "Creative"
        elif 300 <= num < 400:
            return "Business"
        elif 400 <= num < 500:
            return "AI-ML"
        elif 500 <= num < 600:
            return "QA"
        elif 600 <= num < 700:
            return "Education"
        elif 700 <= num < 800:
            return "Science"
        elif 800 <= num < 900:
            return "Leadership"
        elif 900 <= num < 1000:
            return "Legal"
        else:
            return "Other"
    except:
        return "Other"


def split_into_chunks(text: str, max_chars: int = 3000) -> list:
    """텍스트를 청크로 분할"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1
        if current_size + line_size > max_chars and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(line)
        current_size += line_size

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


# ============================================================================
# VectorDB 초기화
# ============================================================================

def init_vectordb():
    """VectorDB 초기화 및 문서 임베딩"""

    print("=" * 60)
    print("  Persona Knowledge Base VectorDB Initializer")
    print("=" * 60)

    # 1. 디렉토리 생성
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/4] VectorDB path: {VECTOR_DB_PATH}")

    # 2. 임베딩 모델 로드
    print("\n[2/4] Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Model loaded: all-MiniLM-L6-v2")

    # 3. ChromaDB 초기화
    print("\n[3/4] Initializing ChromaDB...")
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH),
        settings=Settings(anonymized_telemetry=False)
    )

    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection("persona_knowledge")
        print("  Deleted existing collection")
    except:
        pass

    collection = client.create_collection(
        name="persona_knowledge",
        metadata={"description": "Leaders Decision Assistants Knowledge Base"}
    )
    print("  Created collection: persona_knowledge")

    # 4. 문서 로드 및 임베딩
    print("\n[4/4] Loading and embedding documents...")

    # gpt-knowledge 로드
    print("\n  === GPT Knowledge Files ===")
    knowledge_docs = load_gpt_knowledge()

    # community 페르소나 로드
    print("\n  === Community Personas (detailed) ===")
    persona_docs = load_community_personas()

    # 전체 문서
    all_docs = knowledge_docs + persona_docs

    if not all_docs:
        print("\nNo documents found to index!")
        return

    print(f"\n  Total documents: {len(all_docs)}")

    # 배치 임베딩 및 저장
    print("\n  Embedding and storing...")

    batch_size = 50
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i+batch_size]

        ids = [doc["id"] for doc in batch]
        contents = [doc["content"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]

        # 임베딩 생성
        embeddings = model.encode(contents, show_progress_bar=False).tolist()

        # ChromaDB에 추가
        collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"    Batch {i//batch_size + 1}/{(len(all_docs) + batch_size - 1)//batch_size}: {len(batch)} docs")

    # 완료
    print("\n" + "=" * 60)
    print(f"  VectorDB initialized successfully!")
    print(f"  Total documents: {collection.count()}")
    print(f"  Location: {VECTOR_DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    init_vectordb()
