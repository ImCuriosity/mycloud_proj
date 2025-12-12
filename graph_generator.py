import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import glob
import re

# 설정
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs/graph"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_class_column(value):
    """'류' 컬럼 정제"""
    if pd.isna(value): return "0"
    match = re.search(r'\d+', str(value))
    return match.group(0) if match else "0"

def clean_group_column(value):
    """'유사군' 컬럼 정제 (여러 개일 경우 리스트로 반환)"""
    if pd.isna(value): return ["Unknown_Group"]
    value = str(value)
    # 구분자(콤마, 파이프, 공백)로 분리
    # 예: "G1234 | S5678" -> ["G1234", "S5678"]
    tokens = re.split(r'[|,\s]+', value)
    # 빈 문자열 제거 및 대문자 변환
    codes = [t.strip().upper() for t in tokens if t.strip()]
    if not codes: return ["Unknown_Group"]
    return codes

def load_excel_files():
    all_files = glob.glob(os.path.join(DATA_DIR, "*_DATA.xlsx"))
    df_list = []
    
    print(f"📂 총 {len(all_files)}개의 엑셀 파일을 발견했습니다. 로드 중...")
    
    for filename in tqdm(all_files, desc="Loading Excel"):
        try:
            df = pd.read_excel(filename)
            df.columns = df.columns.str.strip() # 공백 제거

            # 1. 컬럼 매핑
            col_map = {}
            # 상표명칭 (필수)
            if '상표명칭' in df.columns: col_map['상표명칭'] = '상표명칭'
            
            # 류
            for c in ['류', '주요_류', '상품류', 'class']:
                if c in df.columns: col_map[c] = '주요_류'; break
            
            # 유사군 (핵심 추가!)
            for c in ['유사군', '유사군코드', 'similar_group']:
                if c in df.columns: col_map[c] = '유사군'; break

            df.rename(columns=col_map, inplace=True)
            
            if '상표명칭' not in df.columns:
                continue

            # 2. 데이터프레임 생성
            temp_df = pd.DataFrame()
            # 브랜드(Company) 설정
            temp_df['Company_Name'] = df['상표명칭'].fillna("Unknown_Brand")
            # 고유 ID 생성
            temp_df['Trademark_ID'] = df['상표명칭'].fillna("Unknown") + "_" + df.index.astype(str) + "_" + os.path.basename(filename)
            # 류
            temp_df['Class'] = df.get('주요_류', "0")
            # 유사군 (없으면 Unknown 처리)
            temp_df['Group_Raw'] = df.get('유사군', "Unknown_Group")
            
            df_list.append(temp_df)

        except Exception as e:
            print(f"⚠️ {filename} 로드 실패: {e}")
            
    if not df_list: raise ValueError("❌ 로드된 데이터가 없습니다.")
    
    full_df = pd.concat(df_list, ignore_index=True)
    print("🧹 데이터 정제 중 (류 & 유사군)...")
    
    full_df['Class'] = full_df['Class'].apply(clean_class_column)
    
    return full_df

def create_hetero_graph(df):
    print("🕸️ 그래프 데이터 구조 생성 중 (Encoding)...")
    data = HeteroData()

    # 1. 유사군 확장 (Explode)
    # 한 상표에 유사군이 여러 개면 행을 늘려서 처리 (Graph 연결을 위해)
    print("   - 유사군 데이터 확장 중...")
    df['Group_List'] = df['Group_Raw'].apply(clean_group_column)
    # 유사군 별로 행을 쪼갬 (상표 1개 - 유사군 N개 연결)
    df_groups = df.explode('Group_List')[['Trademark_ID', 'Group_List']].dropna()
    df_groups.rename(columns={'Group_List': 'Group_Code'}, inplace=True)

    # 2. 노드 인코딩
    le_company = LabelEncoder()
    le_trademark = LabelEncoder()
    le_class = LabelEncoder()
    le_group = LabelEncoder() # 추가된 인코더

    print("   - 노드 ID 매핑 중...")
    # 문자열 변환
    company_names = df['Company_Name'].astype(str).values
    tm_names = df['Trademark_ID'].astype(str).values
    class_names = df['Class'].astype(str).values
    group_names = df_groups['Group_Code'].astype(str).values
    
    # 핏 & 변환
    # 주의: Trademark ID는 df와 df_groups 양쪽에서 일관성 유지 필요
    le_trademark.fit(tm_names) # 전체 상표 기준 학습
    
    company_ids = le_company.fit_transform(company_names)
    tm_ids_main = le_trademark.transform(tm_names)
    class_ids = le_class.fit_transform(class_names)
    
    # 그룹 데이터 쪽 상표 ID 변환
    tm_ids_group = le_trademark.transform(df_groups['Trademark_ID'].astype(str).values)
    group_ids = le_group.fit_transform(group_names)

    # 노드 메타데이터 저장
    data['company'].num_nodes = len(le_company.classes_)
    data['trademark'].num_nodes = len(le_trademark.classes_)
    data['class'].num_nodes = len(le_class.classes_)
    data['group'].num_nodes = len(le_group.classes_) # 추가

    print(f"    브랜드 노드: {data['company'].num_nodes:,}개")
    print(f"    상표 노드: {data['trademark'].num_nodes:,}개")
    print(f"    류 노드: {data['class'].num_nodes:,}개")
    print(f"    유사군 노드: {data['group'].num_nodes:,}개 (New!)")

    # 3. 엣지 생성
    print("   - 엣지 연결 생성 중...")
    
    # 1) Brand -> Trademark
    src_c = torch.tensor(company_ids, dtype=torch.long)
    dst_t = torch.tensor(tm_ids_main, dtype=torch.long)
    data['company', 'files', 'trademark'].edge_index = torch.stack([src_c, dst_t], dim=0)

    # 2) Trademark -> Class
    src_t = torch.tensor(tm_ids_main, dtype=torch.long)
    dst_cl = torch.tensor(class_ids, dtype=torch.long)
    data['trademark', 'belongs_to', 'class'].edge_index = torch.stack([src_t, dst_cl], dim=0)
    
    # 3) Trademark -> Group (New Edge!)
    src_tg = torch.tensor(tm_ids_group, dtype=torch.long)
    dst_g = torch.tensor(group_ids, dtype=torch.long)
    data['trademark', 'has_code', 'group'].edge_index = torch.stack([src_tg, dst_g], dim=0)

    # 4. 저장
    torch.save({
        'company_classes': le_company.classes_,
        'trademark_classes': le_trademark.classes_,
        'class_classes': le_class.classes_,
        'group_classes': le_group.classes_
    }, os.path.join(OUTPUT_DIR, "label_encoders.pt"))

    return data

if __name__ == "__main__":
    df = load_excel_files()
    graph_data = create_hetero_graph(df)
    
    save_path = os.path.join(OUTPUT_DIR, "graph_data.pt")
    torch.save(graph_data, save_path)
    print(f"\n💾 그래프 재생성 완료 (유사군 포함): {save_path}")