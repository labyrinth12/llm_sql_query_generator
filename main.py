import os
import logging
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.engine.reflection import Inspector
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware



# Load environment variables from .env
load_dotenv()



# ---------------------------
# Configuration
# ---------------------------
class Config:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///retail.db")


# ---------------------------
# Input/Output Schema
# ---------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    sql: str
    result: list


# ---------------------------
# LLM Interface using Gemini
# ---------------------------
class GeminiLLM:
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel("models/gemini-1.5-flash")

    def generate_sql(self, question: str, schema: Dict) -> str:
        schema_text = self._format_schema(schema)
        prompt = self._create_prompt(question, schema_text)
        try:
            response = self.model.generate_content(prompt)
            return self._clean_sql(response.text)
        except Exception as e:
            logging.error(f"Gemini error: {e}")
            return ""

    def _create_prompt(self, question: str, schema: str) -> str:
        return f"""
You are an expert SQL generator.

Given the schema and a natural language question, generate a **SQLite-compatible SQL** query that:
- Encloses all column and table names in double quotes (e.g., "Primary Cement")
- Uses correct spacing and SQL syntax
- Does **not** include comments, explanations, or markdown
- Avoids column aliases unless required
- Avoids malformed quotes or slash characters
- Avoids duplicating column names unless using UNION or UNION ALL properly

{schema}

Question: {question}

Only return the SQL query (no explanation).

SQL:
"""

    def _format_schema(self, schema: Dict) -> str:
        text = "Schema:\n"
        for table, cols in schema.items():
            text += f"Table: {table}\n"
            for col in cols:
                pk = " PRIMARY KEY" if col['primary_key'] else ""
                nn = " NOT NULL" if not col['nullable'] else ""
                text += f"  - {col['name']} ({col['type']}){pk}{nn}\n"
        return text

    def _clean_sql(self, sql: str) -> str:
        return sql.replace("```sql", "").replace("```", "").strip()


# ---------------------------
# Schema Reader
# ---------------------------
class SchemaExtractor:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.metadata = MetaData()

    def extract_schema(self) -> Dict[str, list]:
        self.metadata.reflect(bind=self.engine)
        inspector = Inspector.from_engine(self.engine)
        schema = {}

        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                    "primary_key": col["primary_key"],
                })
            schema[table_name] = columns
        return schema


# ---------------------------
# API Server
# ---------------------------
def create_app():
    config = Config()
    llm = GeminiLLM(config)
    schema_extractor = SchemaExtractor(config.database_url)
    schema = schema_extractor.extract_schema()

    app = FastAPI()

    @app.get("/")
    def root():
        return {"status": "ok", "model": "Gemini SQL"}

    @app.post("/generate", response_model=QueryResponse)
    def generate_sql(request: QueryRequest):
        sql = llm.generate_sql(request.question, schema)
        if not sql:
            raise HTTPException(status_code=500, detail="SQL generation failed")

        try:
            engine = create_engine(config.database_url)
            with engine.connect() as connection:
                result_proxy = connection.execute(text(sql))
                columns = result_proxy.keys()
                rows = result_proxy.fetchall()
                result = [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logging.error(f"DB execution error: {e}")
            raise HTTPException(status_code=400, detail=f"DB execution error: {str(e)}")

        return QueryResponse(sql=sql, result=result)

    return app


app = create_app()
