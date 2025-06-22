# 건설공사 사고 예방 및 대응책 생성 : 한솔데코 시즌3 생성 AI 경진대회(2025.02 ~ 2025.03)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-228B22?style=flat-square)
![SentenceTransformer](https://img.shields.io/badge/SentenceTransformer-006699?style=flat-square)
---

## 1. 프로젝트 소개
 
본 프로젝트는 한솔데코 시즌3 AI 경진대회를 기반으로 진행되었으며, 
건설공사 사고 상황 데이터를 바탕으로 사고 원인을 분석하고 재발방지 대책을 포함한 대응책을 자동으로 생성하는 AI 모델을 목표로 합니다.

---

## 2. 수행 역할

### 사고 보고서 데이터 전처리 및 메타데이터 분류
- ID 기반 규칙을 활용해 `공종`, `사고객체`, `작업프로세스` 등의 결측값을 도메인 지식으로 보정
- `공사종류`, `공종`, `사고객체` 컬럼을 `대분류 / 중분류` 체계로 분할해 정형화
- LLM 및 벡터 검색에서 활용 가능한 형태로 사고 정보를 구조화

```python
train["공종(대분류)"] = train["공종"].str.split(" > ").str[0]
train["사고객체(중분류)"] = train["사고객체"].str.split(" > ").str[1]
for record_id in train[train["사고객체"].isnull()]["ID"]:
    train.loc[train["ID"] == record_id, "사고객체"] = accident_object_fill_values.get(record_id, "기타 > 기타")
```
### 사고 유형에 따른 동적 필터링 기반 RAG 검색 시스템 구축
- 사고 정보(공종, 사고객체 등)에 따라 검색 쿼리를 동적으로 구성
- LangChain을 기반으로 사고 유형에 맞는 유사 문장을 벡터 검색
- 검색된 대응 문장을 직접 반환하거나 LLM 응답 생성의 context로 활용

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
db = FAISS.load_local("vector_db_path", embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.invoke("철근콘크리트공사 / 덤프트럭 / 운반작업 / 철근 낙하")
```
### FAISS를 사용한 벡터 스토어 구축
- 사고 설명 및 대응 문장을 임베딩 후 FAISS로 색인
- `Top-K` 유사 문장을 고속 검색하여 실시간 대응 문장 후보로 활용
- 벡터 기반 유사도 정렬로 유사 사고 탐색 정확도 향상

```python
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("data/accident_cases.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs_split = splitter.split_documents(docs)

db = FAISS.from_documents(docs_split, embedding_model)
db.save_local("vector_db_path")
```

### Ollama 기반 Gemma3:27b 모델을 활용한 대응책 생성
- 검색된 유사 사례 및 사고 설명을 프롬프트에 포함하여 대응 문장 생성
- 로컬 실행 가능한 Gemma3:27b 모델을 Ollama를 통해 호출
- 사고 맥락이 반영된 대응책 생성으로 현장 적용성을 강화

```python
import ollama

prompt = f"""
### 사고 상황:
- 공종: 철근콘크리트공사
- 사고객체: 덤프트럭
- 작업프로세스: 운반작업
- 사고원인: 철근 낙하

### 대응 방안 작성:
"""

response = ollama.chat(model="gemma:3b", messages=[{"role": "user", "content": prompt}])
print(response["message"]["content"])
```

### GPU 가속화 및 대규모 배치 임베딩 처리
- CUDA를 활용한 SentenceTransformer 임베딩 처리 가속
- `tqdm`, 배치 처리 등을 활용해 수천 건 이상의 사고 문장을 효율적으로 벡터화
- OOM 방지를 위한 메모리 최적화 흐름 구성

```python
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer("jhgan/ko-sbert-nli", device="cuda")
embeddings = [model.encode(text) for text in tqdm(data["사고원인"].tolist())]
```
---

## 3. 이슈발생 및 해결과정



---

## 4. 대회 수상팀 코드 분석

---

## 5. 프로젝트 회고
