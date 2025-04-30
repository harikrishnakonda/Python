import openai
import pickle
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
nlp = spacy.load("en_core_web_sm")


class LocationProcessor:
    def __init__(self):
        self.location_db = self._load_embeddings()

    def _load_embeddings(self):
        try:
            with open("location_embeddings.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise Exception("Location embeddings file not found")
        except Exception as e:
            raise Exception(f"Error loading embeddings: {str(e)}")

    def extract_location(self, query):
        doc = nlp(query.lower())
        locations = [ent.text for ent in doc.ents if ent.label_ in ('GPE', 'LOC')]

        prep_pattern = [{'POS': 'ADP'}, {'POS': 'DET', 'OP': '?'}, {'POS': 'PROPN', 'OP': '+'}]
        matcher = spacy.matcher.Matcher(nlp.vocab)
        matcher.add("PREP_LOCATION", [prep_pattern])
        matches = matcher(doc)

        for _, start, end in matches:
            locations.append(doc[start:end].text)

        noun_locations = [chunk.text for chunk in doc.noun_chunks
                          if any(tok.text.lower() in ['city', 'area', 'district'] for tok in chunk)]

        all_candidates = list(set(locations + noun_locations))
        return max(all_candidates, key=lambda x: (
        len(x.split()), any(kw in x.lower() for kw in ['city', 'area']))) if all_candidates else None

    def _get_embedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(response["data"][0]["embedding"])

    def validate_location(self, raw_location):
        if not raw_location:
            return None, 0.0

        query_embedding = self._get_embedding(raw_location)
        max_similarity = -1
        best_match = None

        for location, emb_data in self.location_db.items():
            emb = np.array(emb_data['embedding'] if isinstance(emb_data, dict) else emb_data)
            similarity = cosine_similarity([query_embedding], [emb])[0][0]
            if similarity > max_similarity:
                max_similarity, best_match = similarity, location

        return best_match, max_similarity

    def process_location(self, user_query):
        raw_location = self.extract_location(user_query)
        if not raw_location:
            return None, None

        best_match, confidence = self.validate_location(raw_location)
        return best_match, confidence