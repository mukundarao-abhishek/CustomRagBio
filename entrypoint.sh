#!/bin/bash

# Start the FastAPI server
uvicorn WebAPI:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit server
streamlit run main.py --server.port 8501 --server.enableCORS false

wait -n

exit $?
