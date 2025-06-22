# 건설공사 사고 예방 및 대응책 생성 : 한솔데코 시즌3 생성 AI 경진대회(2025.02 ~ 2025.03)

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
---

## 📈 프로젝트 주요 기능

- **데이터 전처리**: 사고 상황 데이터 정제 및 메타데이터 분류
- **RAG 검색 시스템**: 유사 사고 사례 검색 및 문맥 생성
- **LLM 답변 생성**: Gemma3:27b 모델을 이용한 대응 문구 생성
- **GPU 가속 처리**: RAG 배치 처리 및 임베딩 배치 최적화
- **결과 제출 파일 생성**: 대응 문구 및 임베딩 결과를 CSV로 저장
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

## 🛠️ 프로젝트 기술 스택

Language : 🐍 Python  
Library : 🔗 LangChain | 🧠 SentenceTransformer | 🛠️ FAISS  
Tool : 🐙 GitHub

---

## 🖥️ 시스템 아키텍처 요약

- PDF 텍스트 파일 로드 및 메타데이터 자동 분류
- ko-sbert-sts 기반 텍스트 임베딩 및 FAISS 벡터스토어 구축
- LangChain을 통한 RAG 검색 및 유사 문맥 추출
- Ollama Gemma3:27b 모델을 사용하여 간결한 대응 문구 생성
- 최종 결과를 CSV 파일로 저장

---

## 📊 프로젝트 결과 및 회고

### 결과
- 주어진 사고 상황에 대해 대응책을 생성하는 시스템을 성공적으로 구축했으나, 최종 대회 평가에서는 **자카르 유사도 약 0.49**의 성능을 기록하여 **본선 진출에 실패**했습니다.

### 회고
- **데이터 전처리** 과정에서, 특히 **결측치 처리** 및 **유효 데이터 선별** 부분이 미흡하여 모델 입력 품질에 영향을 주었음을 인지했습니다.
- **프롬프트 엔지니어링**을 보다 정교하게 다듬거나,  
- **LLM(언어 모델)을 직접 파인튜닝**할 수 있었다면, 더욱 일관성 있고 우수한 대응 문구를 생성할 수 있었을 것으로 판단됩니다.
- 또한, **임베딩 품질 개선**이나 **벡터 검색 최적화**를 통해 검색 정확도를 높였어야 했던 점이 아쉬움으로 남습니다.

### 향후 개선 방향
- 사고 상황별 특성에 맞는 **도메인 특화 프롬프트** 설계
- **사전 학습된 LLM 파인튜닝**을 통한 생성 품질 향상
- **전처리 강화** 및 **텍스트 데이터 품질 향상**을 통한 시스템 전반적 성능 개선
