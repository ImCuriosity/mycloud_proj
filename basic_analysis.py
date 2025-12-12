import pandas as pd
import os
import re
import sys
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
import platform

# --- 설정 ---
DATA_DIR = './data/'
OUTPUT_DIR = './outputs/basic/'
LOG_FILE = os.path.join(OUTPUT_DIR, 'analysis_results.txt')

NICE_CLASS_DESC = {
    '1': '화학품', '2': '도료/염료', '3': '화장품/세정제', '4': '산업용 유지', 
    '5': '약제/의약품/위생재', '6': '금속제품', '7': '기계/공작기계', '8': '수공구', 
    '9': '과학/전자/컴퓨터', '10': '의료용 기기', '11': '조명/냉난방', 
    '12': '탈것', '14': '귀금속/시계', '16': '종이/문구', '18': '피혁/가죽', 
    '20': '가구', '21': '가정용구/유리', '25': '의류/신발', 
    '29': '가공식품/육류', '30': '커피/차/제과', '31': '농산물/사료', 
    '35': '광고/경영관리', '36': '보험/금융', '38': '통신', '41': '교육/오락', 
    '42': '과학/기술/IT', '43': '음식점/숙박', '44': '의료/미용',
    '45': '법률/보안', '기타': '기타'
}

# --- 시각화 설정 (한글 폰트 강력 적용) ---
def set_korean_font():
    """
    운영체제별 폰트 파일 경로를 직접 지정하여 한글 깨짐을 방지합니다.
    """
    system_name = platform.system()
    
    if system_name == 'Windows':
        # 윈도우: 맑은 고딕 파일 경로 직접 지정
        font_path = "C:/Windows/Fonts/malgun.ttf"
        if os.path.exists(font_path):
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)
        else:
            plt.rc('font', family='Malgun Gothic') # 파일 없으면 시스템 이름 사용
            
    elif system_name == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
        
    else: # Linux
        plt.rc('font', family='NanumGothic')
    
    plt.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지
    sns.set(font_scale=1.1) 
    # seaborn 스타일 설정 후 폰트 재적용 필요할 수 있음
    sns.set_style("whitegrid")
    
    # Seaborn 설정 후 폰트가 초기화되는 경우가 있어 다시 적용
    if system_name == 'Windows' and os.path.exists(font_path):
        rc('font', family=font_name)
    elif system_name == 'Darwin':
        rc('font', family='AppleGothic')

# pandas 설정
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.precision', 2)

def load_all_data(data_dir):
    all_dfs = []
    if not os.path.exists(data_dir):
        print(f"경로 없음: {data_dir}")
        return pd.DataFrame()

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    country_map = {f: f.split('_')[0].replace('DATA.xlsx', '').replace('.xlsx', '') for f in file_list}

    print("### 1. 데이터 로드 및 통합 시작 ###")
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        country_name = country_map.get(file_name, 'Unknown')
        
        try:
            df = pd.read_excel(file_path)
            df['국가'] = country_name
            all_dfs.append(df)
            print(f"-> 로드 완료: {file_name} (총 {len(df)} 행)")
        except Exception as e:
            print(f"-> 오류 발생: {file_name} 로드 실패 - {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n총 통합 데이터프레임 크기: {len(combined_df)} 행")
        return combined_df
    else:
        print("경고: 로드할 파일이 없습니다.")
        return pd.DataFrame()


def preprocess_data(df):
    print("\n### 2. 데이터 전처리 ###")
    
    df['출원일자'] = pd.to_datetime(df['출원일자'], errors='coerce')
    print(f"-> '출원일자' 컬럼을 datetime 형식으로 변환 완료. (변환 불가한 값: {df['출원일자'].isna().sum()}개)")
    
    df['주요_류'] = df['류'].astype(str).apply(lambda x: x.split('//')[0].strip())
    df['주요_류'] = df['주요_류'].str.extract(r'(\d+)').fillna('기타').astype(str)
    print("-> '류' 컬럼 정제하여 '주요_류' 컬럼 생성 완료.")
    
    # [수정] inplace=True 제거하여 Pandas FutureWarning 해결
    df['상표명칭'] = df['상표명칭'].fillna('(상표명칭 정보 없음)')
    print("-> '상표명칭' 컬럼 결측치 처리 완료.")
    
    return df


def analyze_time_series(df):
    print("\n### 3. 시계열 트렌드 분석 ###")
    
    df_ts = df.dropna(subset=['출원일자']).copy()
    df_ts['출원연도'] = df_ts['출원일자'].dt.year
    
    df_ts = df_ts[(df_ts['출원연도'] >= 2000) & (df_ts['출원연도'] <= 2025)]
    yearly_counts = df_ts.groupby(['출원연도', '국가']).size().reset_index(name='출원수')
    
    print("💡 국가별 출원 건수 Top 5 연도 (터미널 출력 생략)")
    
    # --- 시각화 ---
    try:
        set_korean_font() # 폰트 강제 적용
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=yearly_counts, x='출원연도', y='출원수', hue='국가', marker='o', linewidth=2.5)
        plt.title('국가별 연도별 상표 출원 추이 (2000~)', fontsize=16)
        plt.xlabel('연도')
        plt.ylabel('출원 건수')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '1_time_series_trend.png'), dpi=150)
        plt.close()
        print(f"   [Graph Saved] 1_time_series_trend.png")
    except Exception as e:
        print(f"   [Graph Error] 시계열 그래프 실패: {e}")

    # CAGR 계산 로직 유지
    max_year = yearly_counts['출원연도'].max()
    start_year = max_year - 4 
    cagr_results = []
    for country in yearly_counts['국가'].unique():
        country_data = yearly_counts[yearly_counts['국가'] == country]
        start_count_row = country_data[country_data['출원연도'] == start_year]
        end_count_row = country_data[country_data['출원연도'] == max_year]
        if not start_count_row.empty and not end_count_row.empty:
            beginning_value = start_count_row['출원수'].iloc[0]
            ending_value = end_count_row['출원수'].iloc[0]
            n = max_year - start_year
            if beginning_value > 0:
                cagr = (ending_value / beginning_value) ** (1/n) - 1
                cagr_results.append({'국가': country, f'{start_year}-{max_year} CAGR': f'{cagr * 100:.2f}%'})
    print(f"\n💡 최근 5년 CAGR ({start_year}년 대비 {max_year}년):\n", pd.DataFrame(cagr_results))


def analyze_category(df):
    print("\n### 4. 산업 및 분류 분석 (주요_류 기준) ###")
    
    country_class_counts = df.groupby('국가')['주요_류'].value_counts(normalize=True).mul(100).rename('비중(%)').reset_index()
    country_class_counts['류_설명'] = country_class_counts['주요_류'].astype(str).map(NICE_CLASS_DESC).fillna('기타')
    top_classes = country_class_counts.groupby('국가').head(5).sort_values(by=['국가', '비중(%)'], ascending=[True, False])

    print("💡 국가별 상위 5개 주요_류 비중 (터미널 출력 생략)")
    
    # --- 시각화 ---
    try:
        set_korean_font()
        plt.figure(figsize=(14, 8))
        top_classes['Label'] = top_classes['주요_류'] + '. ' + top_classes['류_설명']
        
        # [수정] hue를 명시하여 Seaborn 경고 해결
        sns.barplot(data=top_classes, x='비중(%)', y='국가', hue='Label', palette='viridis')
        
        plt.title('국가별 Top 5 주요 류(산업) 비중 비교', fontsize=16)
        plt.xlabel('비중 (%)')
        plt.legend(title='주요 류 (NICE Class)', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '2_category_top5.png'), dpi=150)
        plt.close()
        print(f"   [Graph Saved] 2_category_top5.png")
    except Exception as e:
        print(f"   [Graph Error] 카테고리 그래프 실패: {e}")


def analyze_comparison(df):
    print("\n### 5. 글로벌 비교 분석 ###")
    
    diversity_data = []
    for country in df['국가'].unique():
        country_df = df[df['국가'] == country]
        unique_classes = sorted(country_df['주요_류'].unique().tolist())
        diversity_data.append({
            '국가': country,
            '고유_류_개수': len(unique_classes)
        })
        
    diversity_df = pd.DataFrame(diversity_data).sort_values(by='고유_류_개수', ascending=False)
    print("💡 국가별 포트폴리오 다양성:\n", diversity_df)
    
    df['지정상품_개수'] = df['지정상품'].astype(str).apply(lambda x: len(re.split(r'//|,|\n', x)))
    avg_goods = df.groupby('국가')['지정상품_개수'].mean().sort_values(ascending=False).reset_index(name='평균_지정상품_수')
    
    # --- 시각화 ---
    try:
        set_korean_font()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # [수정] hue=country, legend=False 추가하여 Seaborn 경고 해결
        sns.barplot(data=diversity_df, x='국가', y='고유_류_개수', ax=axes[0], hue='국가', palette='Blues_d', legend=False)
        axes[0].set_title('국가별 포트폴리오 다양성 (출원된 류의 종류 수)')
        axes[0].set_ylabel('고유 류 개수')
        
        # [수정] hue=country, legend=False 추가
        sns.barplot(data=avg_goods, x='국가', y='평균_지정상품_수', ax=axes[1], hue='국가', palette='Greens_d', legend=False)
        axes[1].set_title('출원 1건당 평균 지정상품 개수')
        axes[1].set_ylabel('개수')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '3_comparison_diversity_goods.png'), dpi=150)
        plt.close()
        print(f"   [Graph Saved] 3_comparison_diversity_goods.png")
    except Exception as e:
        print(f"   [Graph Error] 비교 분석 그래프 실패: {e}")


def analyze_text(df):
    print("\n### 6. 텍스트 마이닝 (Text Mining & NLP) ###")
    
    df['상표명_길이'] = df['상표명칭'].astype(str).apply(lambda x: len(re.sub(r'\s|\(|\)', '', x)))
    length_summary = df.groupby('국가')['상표명_길이'].agg(['mean', 'median', 'min', 'max']).sort_values(by='mean', ascending=False)
    print("💡 국가별 상표명 길이 요약 통계:\n", length_summary)

    # --- 시각화 ---
    try:
        set_korean_font()
        plt.figure(figsize=(10, 6))
        
        q95 = df['상표명_길이'].quantile(0.95)
        filtered_df = df[df['상표명_길이'] <= q95]
        
        # [수정] hue=country, legend=False 추가
        sns.boxplot(data=filtered_df, x='국가', y='상표명_길이', hue='국가', palette='Set2', legend=False)
        
        plt.title('국가별 상표명 길이 분포 (Outlier 일부 제외)', fontsize=16)
        plt.ylabel('글자 수 (공백제외)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_text_name_length_dist.png'), dpi=150)
        plt.close()
        print(f"   [Graph Saved] 4_text_name_length_dist.png")
    except Exception as e:
        print(f"   [Graph Error] 텍스트 분석 그래프 실패: {e}")
    
    print("\n(키워드 분석 텍스트 출력은 로그 파일 확인 요망)")


# --- 메인 실행 함수 ---
if __name__ == "__main__":
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    set_korean_font() # 시작 전 폰트 설정

    original_stdout = sys.stdout
    string_buffer = StringIO()
    sys.stdout = string_buffer

    try:
        all_data = load_all_data(DATA_DIR)

        if not all_data.empty:
            processed_data = preprocess_data(all_data)

            analyze_time_series(processed_data)
            analyze_category(processed_data)
            analyze_comparison(processed_data)
            analyze_text(processed_data)
            
            print("\n--- 분석 및 시각화 완료 ---")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] 실행 중 오류 발생: {e}")

    finally:
        analysis_results = string_buffer.getvalue()
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(analysis_results)
        
        sys.stdout = original_stdout
        
        print(f"\n✅ 분석 결과 텍스트가 '{LOG_FILE}'에 저장되었습니다.")
        print(f"✅ 시각화 도표들이 '{os.path.abspath(OUTPUT_DIR)}' 폴더에 저장되었습니다.")