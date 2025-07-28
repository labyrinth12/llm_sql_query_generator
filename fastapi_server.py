import logging
import os
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.engine.reflection import Inspector

# Load environment variables from .env
load_dotenv()

# ---------------------------
# Configuration
# ---------------------------
class Config:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///retail.db")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# ---------------------------
# Input/Output Schemas
# ---------------------------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    sql: str
    result: list

class OpenAILLM:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = "gpt-4o"  # or "gpt-3.5-turbo" for cheaper option

    def generate_sql(self, question: str, schema: Dict) -> str:
        schema_text = self._format_schema(schema)
        prompt = self._create_prompt(question, schema_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQLite SQL generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            return self._clean_sql(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"OpenAI error: {e}")
            return ""

    def _create_prompt(self, question: str, schema: str) -> str:
        return f"""
        Given a SQLite schema and a natural language question, output a valid SQLite SQL query that:

        Starts only with SELECT — do not include any explanation or formatting.

        Always includes these columns in the output if they exist:
        "Latitude", "Longitude", "Name of Shop", "Name of Owner", "Taluka Name", "District Name", "Village Name"

        Even if the question asks for grouping, aggregation, or top-N results, you must still return shop-level rows that match those groupings, with the above columns included.

        Never use SELECT *.

        Include only additional columns necessary for answering the question.

        All table and column names must be enclosed in double quotes.

        If the question is about districts, group by "District Name" but join it back to the original table to return individual shop rows with required fields.

        Output only a valid SQL query — no comments, no markdown, no explanation.

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
# Schema Extractor
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
# FastAPI App Factory
# ---------------------------
def create_app():
    config = Config()
    llm = OpenAILLM(config)
    schema_extractor = SchemaExtractor(config.database_url)
    schema = schema_extractor.extract_schema()

    app = FastAPI()

    # Enable CORS if needed
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

# FastAPI app instance
app = create_app()
