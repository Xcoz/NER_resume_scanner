import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

nlp = spacy.load('nlp_model')

with open('data_collector/vacancy.txt', 'r', encoding='utf-8') as vacancy_file:
    vacancy_text = vacancy_file.read()

vacancy_doc = nlp(vacancy_text)

resume_folder = 'data_collector/resume'

tag_weights = {
    "CoreSkills": 1,
    "Skill": 1,
    "Sex": 1,
    "Age": 1,
    "Resides": 1,
    "Nationality": 1,
    "WorkPermit": 1,
    "Relocation": 1,
    "BusinessTrips": 1,
    "Name": 1,
    "Speciality": 1,
    "DesirableTravellingTimeToWork": 1,
    "Employment": 1,
    "WorkShedule": 1,
    "WorkExperience": 1,
    "Education": 1,
    "LanguageSkills": 1,
    "DrivingLicence": 1
}


def get_entity_vacancy_UI(doc):
    with open(doc, 'r', encoding='utf-8') as vacancy_file:
        vacancy_text = vacancy_file.read()
    vacancy_doc = nlp(vacancy_text)
    entity_text = defaultdict(set)
    for ent in vacancy_doc.ents:
        entity_text[ent.label_].add(ent.text.lower())
    return pd.DataFrame({"Label": list(entity_text), "Value": [", ".join(x) for x in entity_text.values()]})


def get_entity_resume_UI(file):
    with open(file, 'r', encoding='utf-8') as resume_file:
        resume_text = resume_file.read()
    resume_doc = nlp(resume_text)
    entity_text = defaultdict(set)
    for ent in resume_doc.ents:
        entity_text[ent.label_].add(ent.text.lower())
    return pd.DataFrame({"Label": list(entity_text), "Value": [", ".join(x) for x in entity_text.values()]})


def get_entity_text(doc):
    entity_text = defaultdict(set)
    for ent in doc.ents:
        entity_text[ent.label_].add(ent.text.lower())
    return entity_text


def calculate_cosine_similarity(vacancy_entities, resume_entities):
    similarity_dict = {}
    for label, vacancy_entity_list in vacancy_entities.items():
        resume_entity_list = resume_entities.get(label, [])
        similarity_list = []

        for vacancy_entity in vacancy_entity_list:
            max_similarity = 0.0

            for resume_entity in resume_entity_list:
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform([vacancy_entity, resume_entity])
                similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

                if similarity > max_similarity:
                    max_similarity = similarity

            similarity_list.append(max_similarity)

        if similarity_list:
            entity_similarity = sum(similarity_list) / len(similarity_list)
            weighted_similarity = min(entity_similarity * tag_weights[label], 1.0)
            similarity_dict[label] = weighted_similarity

    return similarity_dict


def calculate_avg_cosine_similarity(vacancy, resume_list):
    avg_list = []
    with open(vacancy, 'r', encoding='utf-8') as vacancy_file:
        vacancy_text = vacancy_file.read()
        vacancy_doc = nlp(vacancy_text)
        unique_vacancy_entities = get_entity_text(vacancy_doc)

    for resume in resume_list:
        with open(resume, 'r', encoding='utf-8') as resume_file:
            resume_text = resume_file.read()
            resume_doc = nlp(resume_text)
            resume_entity_text = get_entity_text(resume_doc)

            similarity_dict = calculate_cosine_similarity(
                unique_vacancy_entities,
                resume_entity_text
            )

            total_similarity = sum(similarity_dict.values())
            average_similarity = total_similarity / max(1, len(similarity_dict))
            avg_list.append(average_similarity)

    return pd.DataFrame({'Filename': resume_list, 'Similarity': avg_list})
