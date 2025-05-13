import openai
import os
from dotenv import load_dotenv
import logging
import streamlit as st
import json

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

system_messages = {
            "role": "system",
            "content": "You are MatchBot, a helpful assistant specializing in real estate data for Dubai. Provide precise and concise responses, always maintaining a friendly and professional tone."
        }


functions = [
            {
                "name": "fetch_sales_data",
                "description": "Fetch specific real estate data like active sale listings, sold transactions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_type": {"overview"
                            "type": "string",
                            "description": "The type of query being requested (e.g., 'last sold transaction', 'last rental transaction', 'average rental price', 'average sale price', 'location overview')."
                        },
                        "location": {
                            "type": "string",
                            "description": "The location or area specified by the user."
                        },
                        "property_type": {
                            "type": "string",
                            "description": "The type of property (e.g., 'apartment', 'villa')."
                        },
                        "beds": {
                            "type": "integer",
                            "description": "The number of bedrooms in the property."
                        },
                        "transaction_amount": {
                            "type": "number",
                            "description": "The transaction amount in the specified currency."
                        },
                        "timeframe": {
                            "type": "object",
                            "description": "The date of the transaction for sales queries.",
                            "properties": {
                                "start_date": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "The start date of the sales transaction range in YYYY-MM-DD format."
                                },
                                "end_date": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "The end date of the sales transaction range in YYYY-MM-DD format."
                                },
                                "specific_date": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "A specific date for the sales transaction in YYYY-MM-DD format (optional)."
                                }
                            },
                            "anyOf": [
                                {"required": ["start_date"]},
                                {"required": ["end_date"]},
                                {"required": ["specific_date"]}
                            ]
                        },
                        "actual_area_sqm": {
                            "type": "number",
                            "description": "The actual area of the property in square meters."
                        },
                        "property_number": {
                            "type": "string",
                            "description": "The specific unit or property number (e.g., 'Unit 101')."
                        }
                    },
                    "required": ["query_type", "location"]
                }
            }
]

class OpenAIService:
    """
    A class to interact with the OpenAI API for generating SQL queries.
    """
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            st.error("API key not found. Please set 'OPENAI_API_KEY' in your .env file.")
            raise ValueError("API key not found.")


    def prompt(self, messages, temperature=0.1):
            """Handle function calling with GPT-4"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    functions=self.functions,
                    function_call={"name": "fetch_sales_data"},  # Force function call
                    temperature=temperature
                )

                response_message = response["choices"][0]["message"]

                if response_message.get("function_call"):
                    return {
                        "function": response_message["function_call"]["name"],
                        "arguments": json.loads(response_message["function_call"]["arguments"])
                    }

                return None
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                return None

    def format(self, df, query, location=None,temperature=0.1):
        base_prompt = "Analyze this data: {df}\n\nQuery: {query}"
        if location:
            base_prompt += f"\n\nFocus on location: {location}"
        try:
            # Format the messages for the chat input
            messages = [
                {"role": "system", "content": "You are a helpful assistant specializing in data analysis."},
                {"role": "user",
                 "content": f"""Here is the:\n{df} \nQuery: \nPlease respond in plain English based user query {query}
                 check the intent of {query}, if intent is  respond  transaction_amount if user query related to sales ,otherwise
                 respond on these only information about  location name, property_type, beds, transaction_amount, transaction_type and transaction date all data in {df}.
                 mention the transaction type like rental or sales transaction based on {query}.
                 Response should be in plain english with reasoning """}
            ]

            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                timeout=60,
            )

            # Extract and return the content from the response
            return response.choices[0].message['content']
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            st.error("Failed to connect to OpenAI API. Please check your API key and try again.")
            return None

    def process_sales_data(self, arguments):
        """Process sales data request based on function call arguments."""
        try:
            # Implement your actual data fetching logic here
            # This is just a mock implementation
            return {
                'summary': f"Sales data for {arguments.get('location', 'unknown location')}",
                'data': {
                    'average_price': 2500000,
                    'transaction_count': 42,
                    'popular_property_type': 'Apartment'
                }
            }
        except Exception as e:
            logging.error(f"Error processing sales data: {str(e)}")
            return {'error': str(e)}