"""
Start the Gift Card API server.
Run: python run.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="localhost",
        port=7500,
        reload=True,
    )
