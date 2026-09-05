from fastapi import FastAPI

app = FastAPI(
    title="LANALA Banking Recommendation API",
    description="API MLOps de recommandation de produits bancaires",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "LANALA MLOps Recommendation API",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "version": "0.1.0",
    }