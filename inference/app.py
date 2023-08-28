from fastapi import FastAPI
import uvicorn

from routers import inference_router

app = FastAPI()

app.include_router(inference_router.router, prefix="/ml_project_reference")


if __name__ == "__main__":

    uvicorn.run("app:app", reload=True)
