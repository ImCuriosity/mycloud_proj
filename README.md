# MyCloud GNN Project README

## 📂 1. 프로젝트 구조 (Directory Structure)

```plaintext
markcloud_proj/
├── data/                        # [입력] 상표 데이터 엑셀 파일들 (*_DATA.xlsx)
│   ├── 한국_DATA.xlsx
│   ├── 미국_DATA.xlsx
│   └── ...
├── outputs/                     # [출력] 생성된 그래프, 모델, 분석 이미지 저장소
│   ├── graph/                   # graph_data.pt, embeddings.pt 등 저장
│   └── analysis/                # 시장 트렌드 분석 이미지 저장
├── gnn_training_v3_shortcut.py  # GNN 모델 학습 코드
├── graph_generator.py           # 원본 데이터 → 그래프 변환 코드
├── graph_visualization.py       # 그래프 시각화 도구
├── gnn_analysis_final.py        # 통합 AI 분석 실행
├── gnn_korean_expansion.py      # 한국 기업 신사업 추천
├── gnn_korean_competitors.py    # 한국 기업 경쟁사 발굴
├── gnn_korean_gap_analysis.py   # 한국 기업 갭/방어 전략
├── market_trend_analyzer_pro.py # 거시적 시장 트렌드 분석
├── requirements.txt             # 필요한 라이브러리 목록
└── README.md                    # 설명서
```

---

## ⚡ 2. 환경 설정 (Installation)

### 2.1 가상환경 생성 및 실행

```powershell
# 1. 가상환경 생성
python -m venv .venv

# 2. 가상환경 활성화 (Windows PowerShell 기준)\.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

---

### 2.2 필수 라이브러리 설치

**requirements.txt 내용:**

```plaintext
torch
dgl
torch-geometric
numpy
pandas
scikit-learn
networkx
matplotlib
seaborn
openpyxl
tqdm
```

**설치 명령어:**

```powershell
pip install -r requirements.txt
```

---

## 🏗️ 3. 데이터 구축 및 학습 (Build & Train)

### Step 1. 그래프 데이터 생성

```powershell
python graph_generator.py
```

**생성 결과:**
- `./outputs/graph/graph_data.pt` (220만 개 노드 연결)

---

### Step 2. GNN 모델 학습

```powershell
python gnn_training_v3_shortcut.py
```

**생성 결과:**
- `./outputs/graph/dgl_gnn_model_v3.pth`
- `./outputs/graph/dgl_node_embeddings_v3.pt`

---

## 📊 4. AI 분석 및 시각화 (Analysis & Visualization)

학습된 모델을 기반으로 각종 분석 스크립트를 실행합니다.
결과는 `outputs/graph/gnn/`, `outputs/analysis/` 폴더에 저장됩니다.

### 4.1 📈 거시적 시장 트렌드 분석
```powershell
python market_trend_analyzer_pro.py
```

### 4.2 🚀 한국 기업 신사업 예측
```powershell
python gnn_korean_expansion.py
```

### 4.3 ⚔️ 숨겨진 경쟁사 발굴
```powershell
python gnn_korean_competitors.py
```

### 4.4 🛡️ 갭 분석 및 방어 전략
```powershell
python gnn_korean_gap_analysis.py
```

### 4.5 🗺️ 특정 브랜드 생태계 분석
```powershell
python gnn_analysis_final.py
```

---

## 📸 5. 생성되는 분석 이미지 설명

| 파일명 | 설명 |
|--------|------|
| **1_Global_Top_Classes.png** | 전 세계 Top 10 산업군 |
| **1_Country_Top_Classes.png** | 국가별 1등 산업 비교 |
| **1_Korea_Top_Groups.png** | 한국 유사군 Top 10 (스마트폰, 화장품 등) |
| **2_Trends_by_Country.png** | 최근 10년 국가별 트렌드 변화 |
| **3_Promising_Fields_CAGR.png** | 최근 3~4년 유망 분야 Top 5 |
| **4_Seasonality_Trend.png** | 월별 출원 패턴 |

---

필요하시면 프로젝트 소개 문구나 예시 출력까지 포함한 **확장된 버전**도 만들어 드릴 수 있어요!