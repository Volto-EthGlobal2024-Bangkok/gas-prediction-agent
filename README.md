# Backend for Gas Predicting System

To start the server, run following commands:

```bash
docker compose up -d
```

```bash
cd api && npm install && npm run build && npm start
```

# Start model instance

Install `fastapi` and `uvicorn`.

```bash
fastapi dev ./src/routes/env.py
```