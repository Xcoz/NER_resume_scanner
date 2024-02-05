import streamlit as st

from similarity import (get_entity_vacancy_UI,
                        get_entity_resume_UI,
                        calculate_avg_cosine_similarity)


def save_uploadedfile(uploadedfile) -> None:
    with open(uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())


st.title('ЦИАРС.КАДР')

vacancy_uploader = st.file_uploader(label='Загрузите вакансию',
                                    accept_multiple_files=False)
resume_uploader = st.file_uploader(label='Загрузите резюме',
                                   accept_multiple_files=True)


if resume_uploader is not None:
    resume_list = []
    for resume in resume_uploader:
        save_uploadedfile(resume)
        resume_list.append(resume.name)
    resume_selector = st.selectbox('Выбор резюме', resume_list, index=None,
                                   placeholder='Выбор резюме',
                                   label_visibility='collapsed')

col1, col2 = st.columns(2)
with col2:
    if resume_selector is not None:
        df = get_entity_resume_UI(resume_selector)
        st.write('Резюме')
        st.dataframe(data=df, hide_index=True, use_container_width=True)
with col1:
    if vacancy_uploader is not None:
        save_uploadedfile(vacancy_uploader)
        st.write('Вакансия')
        df = get_entity_vacancy_UI(vacancy_uploader.name)
        st.dataframe(data=df, hide_index=True, use_container_width=True)

if resume_uploader and vacancy_uploader is not None:
    df_similarity = calculate_avg_cosine_similarity(vacancy_uploader.name,
                                                    resume_list)
    st.dataframe(data=df_similarity, hide_index=True, use_container_width=True)
