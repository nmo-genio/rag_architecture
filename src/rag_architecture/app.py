from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from rag import query_data

app = FastAPI()

# Initialize counters
question_counter = 0
response_counter = 0

#Enable CORS for local development and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query")
async def ask_mongo(request: Request):
    global question_counter, response_counter
    data = await request.json()
    question = data.get("question", "What is a MongoDB replica set?")
    
    # Increment question counter
    question_counter += 1

    if not question:
        return {"error": "No question provided"}
    
    try:
        answer = query_data(question)
        # Increment response counter on successful response
        response_counter += 1
        return {
            "answer": answer,
            "stats": {
                "questions_asked": question_counter,
                "successful_responses": response_counter
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "stats": {
                "questions_asked": question_counter,
                "successful_responses": response_counter
            }
        }

@app.get("/api/stats")
async def get_stats():
    return {
        "questions_asked": question_counter,
        "successful_responses": response_counter
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)