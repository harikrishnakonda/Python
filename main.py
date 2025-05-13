import pandas as pd
import sqlite3
import logging
import streamlit as st
from gpt import OpenAIService
from locations import LocationProcessor
import re
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_schema():
    try:
        with open("../PythonProject6/Schema_and_description.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Schema file not found. Please ensure 'Schema_and_description.txt' exists.")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the schema file: {str(e)}")
        return ""
# Function to extract relevant schema parts based on query
def extract_relevant_schema(query):
    # You can enhance this logic to extract only the relevant schema portion
    schema_keywords = re.findall(r"\b\w+\b", st.session_state.schema.lower())
    query_keywords = re.findall(r"\b\w+\b", query.lower())

    relevant_parts = []
    for word in query_keywords:
        if word in schema_keywords:
            # Add relevant schema information
            # This is a simple way to check relevance, but you can refine it as per your schema's structure
            relevant_parts.append(word)

    return relevant_parts


# Function to check if a query is relevant to the schema
def is_query_relevant(query):
    try:
        relevant_parts = extract_relevant_schema(query)
        if not relevant_parts:
            return False
        return True
    except Exception as e:
        st.error(f"An error occurred while checking query relevance: {str(e)}")
        return False

def clean_sql_query(query):
    """Remove markdown formatting artifacts from the SQL query."""
    return query.strip().replace("```sql", "").replace("```", "").strip()

def query_db(query):
    """Execute a SQL query on the SQLite database."""
    try:
        with sqlite3.connect('masterdb.db') as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            if query.strip().lower().startswith('select'):
                columns = [description[0] for description in cursor.description]
                results = cursor.fetchall()
                return {"columns": columns, "data": results}
            else:
                conn.commit()
                return {"message": "Query executed successfully."}
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return {"error": f"An error occurred while executing the query: {e}"}

def validate_query_with_db(query):
    """Validate the generated SQL query against the database schema."""
    try:
        with sqlite3.connect('masterdb.db') as conn:
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            return True
    except sqlite3.Error as e:
        logging.error(f"Query validation error: {e}")
        return False

def is_greeting(query):
    """Check if the query is a greeting message."""
    greetings = ["hi", "hello", "hey"]
    return query.lower().strip() in greetings

def main():
    st.title("Conversational SQL Query Bot")

    try:
        gpt = OpenAIService()
        location_processor = LocationProcessor()
    except ValueError:
        st.error("Error initializing GPT service.")
        return

    # Check if schema is already loaded
    if "schema" not in st.session_state:
        st.session_state.schema = load_schema()

    if "conversation" not in st.session_state:
        st.session_state.conversation = [
            {
                "role": "system",
                "content": (
                    """You are MatchBot, a helpful assistant specializing in real estate data for Dubai. Provide precise and concise responses, always maintaining a friendly and professional tone."""
                    "You are an intelligent SQL assistant. Use the following database schema:\n"
                    f""""{load_schema()}\n"
                    "Generate valid SQL queries based on user questions. Validate against the {load_schema()} and database."
                    "Instructions:\n"
                    "If the user query has any Greeting messages like Hi,Hello,hey...then respond to those as Greetings.\n"
                    "based on user query you should extract location name like sadaf1, dubai marina, dubai sports city etc... you should extract only location names and should validate with location_embeddings.pkl and should give best match if location is not present in the user query\n"
                    "location name may be not present because of spelling mistakes, or user query is not based on location name\n"
                    "Include location filters ONLY when provided in the context"
                    "user query like overview, last 6 months rental transaction, average, or some other query there will be no location will be present in the user query\n"
                    "1. Based on the user's question, generate a valid SQL query that matches the database schema. \n"
                    "2. Ensure the query is free of errors, respects schema constraints, and is formatted properly.\n"
                    "3. If the question is ambiguous, make reasonable assumptions and state them clearly.\n"
                    "4. Generate valid SQL queries based on user questions.\n.
                    " Format queries clearly and validate them\n."""

                )
            }
        ]

    user_query = st.text_input("Ask your question:", "")
    if st.button("Submit") and user_query.strip():
        st.session_state.conversation.append({"role": "user", "content": user_query})

        if is_greeting(user_query):
            response = gpt.prompt([{ "role": "user", "content": user_query }], temperature=0.10)
            st.session_state.conversation.append({"role": "assistant", "content": response})
            st.success(response)
            return

        if not is_query_relevant(user_query):
            st.error("The query does not match the schema or contains invalid elements.")
            return

        best_match, confidence = location_processor.process_location(user_query)
        location_context = ""
        # Add location context to conversation if valid
        if best_match and confidence > 0.7:  # Threshold example
            location_context = {
                "role": "system",
                "content": f"User query refers to location: {best_match} (confidence: {confidence:.2f})."
            }
            st.session_state.conversation.append(location_context)

        # Generate SQL
        messages = [
            {"role": "system", "content": st.session_state.conversation[0]["content"]},
            {"role": "user", "content": f"{location_context}\nQuery: {user_query}"}
        ]
        response = gpt.prompt(messages)
        if not response or not response.choices:
            st.error("Failed to generate response.")
            return

        choice = response.choices[0]
        message = choice.message

        # Handle function calls
        if hasattr(message, 'function_call') and message.function_call:
            try:
                function_name = message.function_call.name
                arguments = json.loads(message.function_call.arguments)

                if function_name == "fetch_sales_data":
                    # Call your actual data processing function
                    result = gpt.process_sales_data(arguments)
                    st.success("Here's the real estate data:")
                    st.write(result.get('summary', 'No summary available'))

                    # Add any additional data visualization here
                    if 'data' in result:
                        df = pd.DataFrame(result['data'])
                        st.dataframe(df)

                    return  # Exit after handling function call

            except Exception as e:
                st.error(f"Error processing function call: {str(e)}")
                return

        # If no function call, proceed with SQL processing
        if not message.content:
            st.error("No response generated.")
            return

        sql_query = clean_sql_query(message.content)

        if sql_query:
            st.session_state.conversation.append({"role": "assistant", "content": sql_query})
            # st.success(f"Generated SQL Query:\n{sql_query}")

            if validate_query_with_db(sql_query):
                execution_result = query_db(sql_query)

                if "error" in execution_result:
                    error_message = execution_result["error"]
                    st.error(f"Error: {error_message}")
                    logging.error(f"Execution Error: {error_message}")
                elif "columns" in execution_result and "data" in execution_result:
                    st.success("Query executed successfully! Here are the results:")

                    df = pd.DataFrame(execution_result["data"], columns=execution_result["columns"])

                    response1 = gpt.format(df, user_query)
                    st.write(response1)

                else:
                    st.success(execution_result.get("message", "Query executed successfully."))
            else:
                st.error("The generated query is invalid or does not match the database schema.")
                logging.warning("Invalid query detected during validation.")
        else:
            st.error("No SQL query generated by GPT.")
    # else:
    #     st.error("Failed to get a response from GPT.")

if __name__ == "__main__":
    main()
