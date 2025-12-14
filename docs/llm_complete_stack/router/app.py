from fastapi import FastAPI
app = FastAPI()

@app.post("/route")
def route(query:str):
    if "code" in query:
        return {"expert":"code"}
    return {"expert":"general"}
