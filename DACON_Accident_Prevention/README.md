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

### 결측치 처리

본 프로젝트에서는 단순 결측 제거보다는 **데이터 복원 중심의 수동 보정** 방식을 채택하였다.  
특히 `공종`, `사고객체`, `작업프로세스` 컬럼에 대해서는 사고 ID별로 **의미 있는 값을 사전에 정의**하고,  
해당 ID가 발견되면 그에 맞는 값을 직접 할당하였다. 이는 각 사고 상황의 맥락을 유지하는 데 효과적이었다.

```python
construction_fill_values = {
    "TRAIN_02856": "건축 > 마감공사",
    "TRAIN_13708": "토목 > 토공사",
    # ...
}

for record_id, value in construction_fill_values.items():
    train.loc[train["ID"] == record_id, "공종"] = value
```

`사고객체`, `작업프로세스`에 대해서도 동일한 방식으로 처리하되, 매핑값이 없는 경우에는 `"기타"`로 대체하였다.  
이와 같이 **도메인 기반 수동 보정**은 정제된 입력 품질을 확보하면서 데이터 손실을 방지할 수 있었다.

반면 `인적사고`, `물적사고`, `사고원인` 등은 비교적 일반적인 결측 처리 방식을 적용하였다.  
텍스트 분석 모델 입력 시 오류가 발생하지 않도록 하기 위해, 다음과 같은 기본값을 지정하였다.

```python
train["인적사고"].fillna("없음", inplace=True)
train["물적사고"].fillna("없음", inplace=True)
train["사고원인"].fillna("기타", inplace=True)
```

### 사고 보고서 데이터 전처리 및 메타데이터 분류
- 사고 보고서 파일명을 기반으로 공사유형, 작업유형, 사고유형, 사고객체를 자동 분류할 수 있는 메타데이터 사전 구성
- 질문 내 포함된 키워드에 따라 동적으로 필터링 조건을 생성해 검색 성능을 높임

| 메타데이터 항목     | 분류 항목       | 키워드 예시                                |
|--------------------|----------------|--------------------------------------------|
| construction_type  | 건축           | 건축물, 건설공사, 건설기계, 건설현장, 건설업체 |
| construction_type  | 토목           | 교량, 터널, 도로, 철도, 항만, 하천         |
| construction_type  | 조경           | 조경, 수목, 식재                            |
| construction_type  | 설비           | 설비, 플랜트, 시스템, 기계                 |
| construction_type  | 기타           | (미포함 시 기타)                           |

### RAG 기반 검색 시스템 구축
- LangChain 벡터 검색 시스템 위에 메타데이터 필터링을 추가해 유사 사례의 정밀도를 향상시킴
- 질문별 사고 맥락에 맞는 사례만 걸러내어 문맥 신뢰도 확보

```python
rag_chain = (
    {
        'context': lambda inputs: "\n\n".join([res['section'] for res in search_similar_sections(
            inputs['question'], vectorstore, filters=get_dynamic_filters(inputs['question']), k=3
        )]),
        'question': itemgetter("question")
    }
    | prompt 
    | llm
    | StrOutputParser()
)
```

### FAISS 벡터 스토어 구축 및 GPU 임베딩 처리
- `jhgan/ko-sbert-sts` 모델을 활용해 문서를 임베딩하고 FAISS로 저장
- 배치 기반 인코딩과 GPU 연산을 통해 수천 개 문서도 빠르게 처리 가능

```python
model = SentenceTransformer("jhgan/ko-sbert-sts").to(device)
embeddings = model.encode(batch, device=device)
```

### Ollama 기반 Gemma3 모델을 활용한 대응책 생성
- 사고 질문에 대해 유사 사례를 context로 구성한 후, Gemma3 모델을 통해 간결한 대응 문장을 생성
- 최대 100토큰 제한, 형식 통일, 불필요한 수식어 제거 등 평가 기준에 맞춘 응답 형식 제어

```python
llm = ChatOllama(model='gemma3:27b', temperature=0.0)
result = llm.invoke({"context": context, "question": question})
```

---

## 3. 이슈발생 및 해결과정



---

## 4. 대회 수상팀 코드 분석

1등 팀은 `사고원인` 컬럼을 문장 단위로 임베딩한 후, cosine similarity를 활용하여 유사 사고를 탐색하고  
해당 사고의 대응 문장을 직접 참조하여 예측값을 생성하는 방식으로 접근하였습니다.  
이러한 구조는 자카르 유사도 평가 방식에 매우 적합하며, 문장 내 핵심 단어의 일치를 통해 높은 점수를 얻을 수 있었습니다.

또한, 문장 길이의 평균과 중앙값을 분석하여 응답 문장의 길이를 일정 수준으로 유지하려는 정제 작업이 이루어졌으며,  
복합 피처가 아닌 단일 문장 기준의 임베딩을 통해 보다 직관적이고 정확한 유사도 매칭이 가능하였습니다.

| 비교 항목 | 수상팀 | 본 프로젝트 |
|-----------|--------|-------------|
| 임베딩 기준 | 사고원인 (문장) | 공종, 사고객체, 작업프로세스, 사고원인 (복합) |
| 유사도 계산 | cosine similarity | 벡터 기반 검색 (RAG) |
| 응답 방식 | 기존 문장 참조 및 조합 | LLM 생성 또는 검색 결과 활용 |
| 평가 대응 전략 | 자카르 유사도 최적화 | 의미 기반 응답 중심 구조 |

이처럼 수상팀은 문장 기반 검색과 참조 방식을 통해 평가 지표에 직접적으로 대응한 반면,  
본 프로젝트는 복합 정보를 활용한 유연한 생성 구조를 기반으로 하여, 실무 활용성과 확장성을 고려한 접근이었다고 판단됩니다.

## 5. 프로젝트 회고

본 프로젝트에서는 사고 유형을 보다 현실적으로 반영하기 위해,  
`공종`, `사고객체`, `작업프로세스`, `사고원인` 등의 정보를 분리하고 구조화한 후  
LangChain 기반 RAG 검색 구조와 Ollama 기반 LLM(Gemma3:27b)을 결합하여 대응 문장을 생성하였습니다.

이 과정에서 가장 중점을 둔 부분은 **결측치 처리**입니다. 단순한 결측 제거가 아닌, 사고 ID에 따라 도메인 지식을 반영한 수동 보정 방식을 채택하여  
데이터의 누락을 최소화하고 정보의 일관성을 유지하고자 하였습니다.

```python
construction_fill_values = {
    "TRAIN_02856": "건축 > 마감공사",
    ...
}
for record_id, value in construction_fill_values.items():
    train.loc[train["ID"] == record_id, "공종"] = value

train["인적사고"].fillna("없음", inplace=True)
train["사고원인"].fillna("기타", inplace=True)
```

사고 데이터를 구조적으로 정제하여 안정적인 입력 기반을 마련하였으나,  
자카르 유사도와 같은 정량 평가 지표에 직접 대응하기에는 상대적으로 불리한 측면도 있었습니다.  
생성형 응답은 의미적으로 풍부하고 자연스러웠지만, 기존 문장을 참조하는 방식에 비해 평가 점수에서는 낮게 나올 수 있음을 확인하였습니다.

이러한 경험을 통해 향후에는 **평가지표에 따라 생성형과 검색형 전략을 유연하게 조합**할 수 있는 구조를 설계하는 것이 중요하다는 점을 크게 느꼈습니다.

> 복합 사고 요소를 고려한 구조적 설계는 실무 활용에 적합하였지만,  
> 정량 평가 중심의 환경에서는 참조 기반 응답 방식이 더욱 효과적일 수 있다는 점을 체감하였습니다.
