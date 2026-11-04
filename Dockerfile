FROM python:3.11-slim AS build
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt
FROM python:3.11-slim
RUN useradd -m app
COPY --from=build /deps /usr/local
WORKDIR /app
COPY backend/ .
COPY public ./public
USER app
ENV PORT=10000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
