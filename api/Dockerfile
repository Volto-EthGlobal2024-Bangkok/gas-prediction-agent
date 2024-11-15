FROM node:20-bookworm-slim as builder

# ENV HOST=0.0.0.0
# ENV PORT=3000
# ENV ORIGIN="https://www.sila.zone"

WORKDIR /app

COPY . .

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y openssl \
    && npm install --frozen-lockfile --omit=dev \
    # && ./node_modules/.bin/prisma generate \
    && npm run build

EXPOSE 3000

CMD [ "node", "-r", "dotenv/config", "./api/dist/index.js" ]