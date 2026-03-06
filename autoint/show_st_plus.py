# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import os

# import joblib
# from autointmlp import AutoIntMLPModel, predict_model
# from tensorflow.keras.models import load_model

# # streamlit run show_st.py 

# @st.cache_resource
# def load_data():
#     '''
#     앱에서 보여줄 필요 데이터를 가져오는 함수입니다.
#     - 사용자, 영화, 평점 데이터를 가져옵니다.
#     - 앞서 저장된 모델도 불러오고 구현해둡니다.
#     '''
#     project_path = os.path.abspath(os.getcwd())
#     data_dir_nm = 'data'
#     movielens_dir_nm = 'ml-1m'
#     model_dir_nm = 'model'
#     data_path = f"{project_path}/{data_dir_nm}"
#     model_path = f"{project_path}/{model_dir_nm}"
#     field_dims = np.load(f'{data_path}/field_dims.npy')
#     dropout= 0.4
#     embed_dim= 16
    
#     ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
#     movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
#     user_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')


#     model = AutoIntMLPModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
#                              l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001)
    

    
#     model(tf.constant([[0] * len(field_dims)], dtype=tf.int64))

#     model.load_weights(f'{model_path}/autoIntMLP_model_weights.weights.h5') 
#     label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    
#     return user_df, movies_df, ratings_df, model, label_encoders


# def get_user_seen_movies(ratings_df):
#     '''
#     사용자가 과거에 보았던 영화 리스트를 가져옵니다.
#     '''
#     user_seen_movies = ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()
#     return user_seen_movies

# def get_user_non_seed_dict(movies_df, user_df, user_seen_movies):
#     '''
#     사용자가 보지 않았던 영화 리스트를 가져옵니다.
#     '''
#     unique_movies = movies_df['movie_id'].unique()
#     unique_users = user_df['user_id'].unique()
#     user_non_seen_dict = dict()

#     for user in unique_users:
#         user_seen_movie_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
#         user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_movie_list))
#         user_non_seen_dict[user] = user_non_seen_movie_list
        
#     return user_non_seen_dict


# def get_user_info(user_id):
#     '''
#     사용자 정보를 가져옵니다.
#     '''
#     return users_df[users_df['user_id'] == user_id]

# def get_user_past_interactions(user_id):
#     '''
#     사용자 평점 데이터 중 4점 이상(선호했다는 정보)만 가져옵니다. 
#     '''
#     return ratings_df[ (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')


# def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
#     '''
#     아래와 같은 순서로 추천 결과를 가져옵니다.
#     1. streamlit에서 입력 받은 타겟 월, 연도, 사용자 정보를 받아옴
#     2. 사용자가 보지 않았던 정보 추출
#     3. model input으로 넣을 수 있는 형태로 데이터프레임 구성
#     4. label encoder 적용해 모델에 넣을 준비
#     5. 모델 predict 수행
#     '''
#     user_non_seen_movie = user_non_seen_dict.get(user)
#     user_id_list = [user for _ in range(len(user_non_seen_movie))]
#     r_decade = str(r_year - (r_year % 10)) + 's'
    
#     user_non_seen_movie = pd.merge(pd.DataFrame({'movie_id':user_non_seen_movie}), movies_df, on='movie_id')
#     user_info = pd.merge(pd.DataFrame({'user_id':user_id_list}), user_df, on='user_id')
#     user_info['rating_year'] = r_year
#     user_info['rating_month'] = r_month
#     user_info['rating_decade'] = r_decade
    
#     merge_data = pd.concat([user_non_seen_movie, user_info], axis=1)
#     merge_data.fillna('no', inplace=True)
#     merge_data = merge_data[['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']]
    
#     for col, le in label_encoders.items():
#         merge_data[col] = le.fit_transform(merge_data[col])
    
#     recom_top = predict_model(model, merge_data)
#     # 추천 중 영화 id에 해당되는 부분만 가져옴
#     recom_top = [r[0] for r in recom_top]
#     # 원본 영화 id로 변환
#     origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    
#     # 영화 리스트를 매칭시켜 가져옴
#     return movies_df[movies_df['movie_id'].isin(origin_m_id)]

# # 데이터 로드
# users_df, movies_df, ratings_df, model, label_encoders = load_data()
# user_seen_movies = get_user_seen_movies(ratings_df)
# user_non_seen_dict = get_user_non_seed_dict(movies_df, users_df, user_seen_movies)

# # 타이틀
# st.title("영화 추천 결과 살펴보기")

# st.header("사용자 정보를 넣어주세요.")
# user_id = st.number_input("사용자 ID 입력", min_value=users_df['user_id'].min(), max_value=users_df['user_id'].max(), value=users_df['user_id'].min())
# r_year = st.number_input("추천 타겟 연도 입력", min_value=ratings_df['rating_year'].min(), max_value=ratings_df['rating_year'].max(), value=ratings_df['rating_year'].min())
# r_month = st.number_input("추천 타겟 월 입력", min_value=ratings_df['rating_month'].min(), max_value=ratings_df['rating_month'].max(), value=ratings_df['rating_month'].min())
 

# # streamlit run show_st.py --client.showErrorDetails=false
# if st.button("추천 결과 보기"):
#     st.write("사용자 기본 정보")
#     user_info = get_user_info(user_id)
#     st.dataframe(user_info)

#     st.write("샤용자가 과거에 봤던 이력(평점 4점 이상)")
#     user_interactions = get_user_past_interactions(user_id)
#     st.dataframe(user_interactions)

#     st.write("추천 결과")
#     recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders)
    
#     st.dataframe(recommendations)


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os

import joblib
from autointmlp import AutoIntMLPModel, predict_model
from tensorflow.keras.models import load_model

# ==========================================
#[UI 개선] 페이지 기본 설정 (가장 상단에 위치해야 합니다)
# ==========================================
st.set_page_config(
    page_title="영화 맞춤 추천 시스템",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 기존 로직 및 함수 (수정하지 않음)
# ==========================================
@st.cache_resource
def load_data():
    '''
    앱에서 보여줄 필요 데이터를 가져오는 함수입니다.
    - 사용자, 영화, 평점 데이터를 가져옵니다.
    - 앞서 저장된 모델도 불러오고 구현해둡니다.
    '''
    project_path = os.path.abspath(os.getcwd())
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = f"{project_path}/{data_dir_nm}"
    model_path = f"{project_path}/{model_dir_nm}"
    field_dims = np.load(f'{data_path}/field_dims.npy')
    dropout= 0.4
    embed_dim= 16
    
    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
    user_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')

    model = AutoIntMLPModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
                             l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001)
    
    model(tf.constant([[0] * len(field_dims)], dtype=tf.int64))

    model.load_weights(f'{model_path}/autoIntMLP_model_weights.weights.h5') 
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    
    return user_df, movies_df, ratings_df, model, label_encoders


def get_user_seen_movies(ratings_df):
    '''
    사용자가 과거에 보았던 영화 리스트를 가져옵니다.
    '''
    user_seen_movies = ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()
    return user_seen_movies

def get_user_non_seed_dict(movies_df, user_df, user_seen_movies):
    '''
    사용자가 보지 않았던 영화 리스트를 가져옵니다.
    '''
    unique_movies = movies_df['movie_id'].unique()
    unique_users = user_df['user_id'].unique()
    user_non_seen_dict = dict()

    for user in unique_users:
        user_seen_movie_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
        user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_movie_list))
        user_non_seen_dict[user] = user_non_seen_movie_list
        
    return user_non_seen_dict


def get_user_info(user_id):
    '''
    사용자 정보를 가져옵니다.
    '''
    return users_df[users_df['user_id'] == user_id]

def get_user_past_interactions(user_id):
    '''
    사용자 평점 데이터 중 4점 이상(선호했다는 정보)만 가져옵니다. 
    '''
    return ratings_df[ (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')


def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
    '''
    아래와 같은 순서로 추천 결과를 가져옵니다.
    1. streamlit에서 입력 받은 타겟 월, 연도, 사용자 정보를 받아옴
    ...
    '''
    user_non_seen_movie = user_non_seen_dict.get(user)
    user_id_list =[user for _ in range(len(user_non_seen_movie))]
    r_decade = str(r_year - (r_year % 10)) + 's'
    
    user_non_seen_movie = pd.merge(pd.DataFrame({'movie_id':user_non_seen_movie}), movies_df, on='movie_id')
    user_info = pd.merge(pd.DataFrame({'user_id':user_id_list}), user_df, on='user_id')
    user_info['rating_year'] = r_year
    user_info['rating_month'] = r_month
    user_info['rating_decade'] = r_decade
    
    merge_data = pd.concat([user_non_seen_movie, user_info], axis=1)
    merge_data.fillna('no', inplace=True)
    merge_data = merge_data[['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']]
    
    for col, le in label_encoders.items():
        merge_data[col] = le.fit_transform(merge_data[col])
    
    recom_top = predict_model(model, merge_data)
    # 추천 중 영화 id에 해당되는 부분만 가져옴
    recom_top = [r[0] for r in recom_top]
    # 원본 영화 id로 변환
    origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    
    # 영화 리스트를 매칭시켜 가져옴
    return movies_df[movies_df['movie_id'].isin(origin_m_id)]

# 데이터 로드 (로딩 스피너 적용)
with st.spinner('데이터와 AI 모델을 불러오는 중입니다... ⏳'):
    users_df, movies_df, ratings_df, model, label_encoders = load_data()
    user_seen_movies = get_user_seen_movies(ratings_df)
    user_non_seen_dict = get_user_non_seed_dict(movies_df, users_df, user_seen_movies)


# ==========================================
# [UI 개선] 메인 화면 및 사이드바 레이아웃 구성
# ==========================================

# 메인 타이틀
st.title("🎬 딥러닝 기반 영화 맞춤 추천 시스템")
st.markdown("사용자의 과거 시청 이력과 프로필을 분석하여 **가장 취향에 맞는 영화 Top 10**을 추천해 드립니다.")
st.divider()

# 사이드바 (사용자 입력 폼)
with st.sidebar:
    st.header("⚙️ 추천 설정")
    st.markdown("추천을 받을 **사용자 ID**와 **기준 시점**을 선택해주세요.")
    
    st.write("---")
    user_id = st.number_input("👤 사용자 ID 입력", min_value=users_df['user_id'].min(), max_value=users_df['user_id'].max(), value=users_df['user_id'].min())
    r_year = st.number_input("📅 추천 타겟 연도", min_value=ratings_df['rating_year'].min(), max_value=ratings_df['rating_year'].max(), value=ratings_df['rating_year'].min())
    r_month = st.number_input("🗓️ 추천 타겟 월", min_value=ratings_df['rating_month'].min(), max_value=ratings_df['rating_month'].max(), value=ratings_df['rating_month'].min())
    st.write("---")
    
    # 사이드바 버튼 (use_container_width=True 로 폭 꽉차게 예쁘게)
    submit_button = st.button("🚀 AI 추천 결과 보기", use_container_width=True, type="primary")

# 버튼 클릭 시 결과 출력 영역
if submit_button:
    # 1. 데이터를 불러오는 동안 로딩 애니메이션 표시
    with st.spinner('AI가 수만 개의 영화 중 사용자 맞춤 영화를 분석하고 있습니다... 🤖'):
        user_info = get_user_info(user_id)
        user_interactions = get_user_past_interactions(user_id)
        recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders)
    
    # 2. 탭(Tabs)을 사용하여 화면을 깔끔하게 분할 (스크롤 압박 방지)
    tab1, tab2, tab3 = st.tabs(["✨ AI 추천 결과", "👤 사용자 프로필", "🎞️ 과거 선호 이력 (4점 이상)"])
    
    with tab1:
        st.subheader(f"🎉 사용자 {user_id}님을 위한 맞춤 추천 영화")
        st.markdown("모델 예측 결과, 아직 보지 않은 영화 중 가장 선호도가 높을 것으로 예상되는 Top 10 입니다.")
        # 데이터프레임 가로폭 채우기 및 인덱스 숨기기 적용
        st.dataframe(recommendations, use_container_width=True, hide_index=True)
        
    with tab2:
        st.subheader("👤 사용자 기본 정보")
        st.dataframe(user_info, use_container_width=True, hide_index=True)
        
    with tab3:
        st.subheader(f"🎞️ 사용자 {user_id}님이 높게 평가한 영화 이력")
        st.markdown(f"과거에 **4점 이상**의 높은 평점을 남긴 영화는 총 **{len(user_interactions)}편** 입니다.")
        st.dataframe(user_interactions, use_container_width=True, hide_index=True)

else:
    # 버튼을 누르기 전 메인 화면 안내문
    st.info("👈 왼쪽 사이드바에서 사용자 정보를 입력하고 **'🚀 AI 추천 결과 보기'** 버튼을 클릭해주세요!")