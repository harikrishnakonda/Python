## real_estate_bot.py
import re
import logging
from sqlalchemy import create_engine, text
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import spacy
from tenacity import retry, wait_exponential, stop_after_attempt

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
engine = create_engine(os.getenv("DATABASE_URI"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocationProcessor:
    def __init__(self, embeddings_path="location_embeddings.pkl"):
        self.embeddings_path = embeddings_path
        self.similarity_threshold = 0.7
        self.nlp = spacy.load("en_core_web_md")
        self.location_db = self._load_embeddings()

    def _load_embeddings(self):
        try:
            with open(self.embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
                return {k: np.array(v['embedding']) if isinstance(v, dict) else v
                        for k, v in embeddings.items()}
        except Exception as e:
            logger.error(f"Embedding load failed: {str(e)}")
            return {}

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def validate_location(self, location):
        try:
            if not location:
                return None, 0.0

            query_emb = self._get_embedding(location.lower())
            best_match, max_similarity = None, 0.0

            for loc_name, loc_emb in self.location_db.items():
                if isinstance(loc_emb, dict):
                    loc_emb = np.array(loc_emb.get('embedding', []))

                similarity = cosine_similarity([query_emb], [loc_emb])[0][0]
                if similarity > max_similarity:
                    best_match, max_similarity = loc_name, similarity

            return best_match if max_similarity >= self.similarity_threshold else None, max_similarity
        except Exception as e:
            logger.error(f"Location validation failed: {str(e)}")
            return None, 0.0

    def _get_embedding(self, text):
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return np.array(response["data"][0]["embedding"])


class RealEstateBot:
    def __init__(self):
        self.location_processor = LocationProcessor()
        self.conversation_history = []

    def chat(self, user_input):
        try:
            # Process location
            raw_location = self.extract_location(user_input)
            location, confidence = self.location_processor.validate_location(raw_location)

            if not location:
                return "Location not recognized. Please try again."

            # Generate SQL with default parameters
            sql_query, params = self.generate_sql(
                validated_location=location,
                user_input=user_input
            )

            # Execute query
            result = pd.read_sql(sql_query, engine, params=params)

            # Generate analysis
            analysis = self.generate_analysis(result, user_input)

            return self.format_response(result, analysis, params)

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return f"Error processing request: {str(e)}"

    def generate_sql(self, validated_location, user_input):
        # Use actual database column names
        SCHEMA = """[Database Schema]
        - sales_transactions: 
          id, transaction_date, bayut_property_type_name_en, beds, 
          bayut_leaf_location_name_en, transaction_amount, actual_area_sqm, property_number"""

        # Correct date calculation (2 years back)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

        params = {
            'location': validated_location,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'property_type': None,
            'beds': None
        }

        # Extract parameters with proper case handling
        params.update(self.extract_parameters(user_input))

        base_query = text(f"""
        SELECT * FROM sales_transactions 
        WHERE bayut_leaf_location_name_en = :location
        AND date_transaction_nk BETWEEN :start_date AND :end_date
        {"AND LOWER(bayut_property_type_name_en) = LOWER(:property_type)" if params.get('property_type') else ""}
        {"AND beds = :beds" if params.get('beds') else ""}
        ORDER BY date_transaction_nk DESC
        LIMIT 10
        """)

        return base_query, params

    def extract_parameters(self, text):
        params = {}
        text_lower = text.lower()

        # Property type mapping
        property_map = {
            'apartment': ['apartment', 'flat', 'unit'],
            'villa': ['villa', 'house'],
            'townhouse': ['townhouse', 'town home'],
            'office': ['office', 'commercial space']
        }

        for prop_type, keywords in property_map.items():
            if any(kw in text_lower for kw in keywords):
                params['property_type'] = prop_type.capitalize()
                break

        # Bedrooms extraction
        bed_match = re.search(r'\b(\d+)\s+(bed|beds|bedroom)\b', text, re.IGNORECASE)
        if bed_match:
            params['beds'] = int(bed_match.group(1))

        return params

    def generate_analysis(self, df, query):
        prompt = f"""Analyze this real estate data:
        {df.head(10).to_markdown()}

        User query: {query}

        Provide insights focusing on:
        - Price trends
        - Location patterns
        - Property type variations"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a real estate data analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message['content']
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return "Analysis unavailable"

    def format_response(self, data, analysis, params):
        return f"""
        Real Estate Analysis Report
        ---------------------------
        Location: {params['area']}
        Timeframe: {params['start_date']} to {params['end_date']}
        {f"Property Type: {params.get('property_type', 'All')}"}
        {f"Bedrooms: {params.get('beds', 'Any')}"}

        Data Preview:
        {data.head().to_string()}

        AI Analysis:
        {analysis}
        """

    def extract_location(self, text):
        doc = self.location_processor.nlp(text.lower())
        for ent in doc.ents:
            if ent.label_ in ('GPE', 'LOC', 'FAC'):
                return ent.text
        return text


def main():
    bot = RealEstateBot()
    print("Real Estate Data Bot - Type 'exit' to quit")

    while True:
        user_input = input("\nYour query: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        response = bot.chat(user_input)
        print("\n" + response)


if __name__ == '__main__':
    main()