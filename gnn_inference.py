import torch
import torch.nn.functional as F
import numpy as np
import os

# ==========================================
# ⚙️ 설정
# ==========================================
EMBEDDING_PATH = "./outputs/graph/dgl_node_embeddings.pt"
ENCODER_PATH = "./outputs/graph/label_encoders.pt"
GRAPH_PATH = "./outputs/graph/graph_data.pt" # 이미 진출한 분야 확인용

def load_resources():
    print("🔄 데이터 로드 중...")
    
    # 1. 학습된 임베딩 (GPU에서 학습했어도 CPU로 로드)
    embeddings = torch.load(EMBEDDING_PATH, map_location='cpu')
    
    # 2. 이름 인코더 (ID -> 텍스트 변환)
    encoders = torch.load(ENCODER_PATH, weights_only=False)
    
    # 3. 원본 그래프 (이미 보유한 상표 확인용)
    # PyTorch 버전에 따라 호환성 처리
    try:
        graph_data = torch.load(GRAPH_PATH, weights_only=False)
    except TypeError:
        graph_data = torch.load(GRAPH_PATH)
        
    print("✅ 리소스 로드 완료!")
    return embeddings, encoders, graph_data

def get_recommendations(company_name, embeddings, encoders, graph_data, top_k=5):
    """
    벡터 유사도(Cosine Similarity) 기반 추천 로직
    """
    comp_names = encoders['company_classes']
    class_names = encoders['class_classes']
    
    # 1. 기업 ID 찾기
    try:
        target_idx = np.where(comp_names == company_name)[0][0]
    except IndexError:
        print(f"❌ '{company_name}' 기업을 찾을 수 없습니다.")
        return

    # 2. 기업 벡터 가져오기
    # embeddings 딕셔너리에서 'company' 키의 값 중 target_idx 행
    company_vec = embeddings['company'][target_idx].unsqueeze(0) # [1, 64]
    
    # 3. 모든 류(Class) 벡터 가져오기
    class_vecs = embeddings['class'] # [Num_Classes, 64]
    
    # 4. 코사인 유사도 계산
    # (내 벡터와 가장 방향이 비슷한 벡터 찾기)
    similarity = F.cosine_similarity(company_vec, class_vecs)
    
    # 5. 이미 진출한 분야 제외하기 (필터링)
    # 그래프에서 해당 기업이 이미 연결된 상표 -> 그 상표가 속한 류 찾기
    c_t_edge = graph_data['company', 'files', 'trademark'].edge_index
    t_c_edge = graph_data['trademark', 'belongs_to', 'class'].edge_index
    
    # 내 상표들
    my_tm_mask = (c_t_edge[0] == target_idx)
    my_tm_ids = c_t_edge[1][my_tm_mask]
    
    # 내 류들 (보유 중인)
    # t_c_edge[0] 가 my_tm_ids에 포함되는지 확인
    # (간단하게 구현: 반복문 없이 마스킹)
    if len(my_tm_ids) > 0:
        # 1. my_tm_ids가 CPU 텐서인지 확인
        my_tm_ids = my_tm_ids.cpu()
        # 2. t_c_edge 전체 탐색 (데이터가 아주 크지 않다면 가능)
        mask = torch.isin(t_c_edge[0], my_tm_ids)
        my_class_ids = t_c_edge[1][mask].unique()
        
        # 이미 가진 류의 점수는 -무한대로 설정하여 추천 제외
        similarity[my_class_ids] = -9999.0
        
        # 현재 진출 현황 출력
        current_classes = [class_names[i] for i in my_class_ids[:5]] # 5개만 표기
        print(f"\n🏢 기업명: {company_name}")
        print(f"ℹ️ 현재 진출 분야({len(my_class_ids)}개): {', '.join(map(str, current_classes))} ...")

    # 6. 상위 K개 추천
    top_scores, top_indices = torch.topk(similarity, top_k)
    
    print(f"\n💡 [AI 추천] {company_name} 님을 위한 사업 확장 유망 분야")
    print("-" * 50)
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
        cls_name = class_names[idx.item()]
        print(f"   🏆 {rank+1}위: 류(Class) {cls_name:<5} (유사도: {score:.4f})")
    print("-" * 50)

if __name__ == "__main__":
    embeddings, encoders, graph_data = load_resources()
    
    # 1위 기업 이름 가져오기 (예시)
    # encoders['company_classes']에 있는 아무 기업이나 넣으셔도 됩니다.
    first_company = encoders['company_classes'][0]
    
    # 추천 실행
    get_recommendations(first_company, embeddings, encoders, graph_data)
    
    # 원하시는 기업 이름을 직접 넣으셔도 됩니다.
    # get_recommendations("SAMSUNG ELECTRONICS CO., LTD.", embeddings, encoders, graph_data)