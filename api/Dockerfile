FROM node:16-alpine as builder

RUN apk update && apk add yarn curl bash python3 g++ make && rm -rf /var/cache/apk/*
# Install node-prune
RUN curl -sfL https://install.goreleaser.com/github.com/tj/node-prune.sh | bash -s -- -b /usr/local/bin

WORKDIR /app

COPY . .

RUN rm -rf /app/api/node_modules
WORKDIR /app/api

RUN npm i -g ts-project-bundle

# Install dependencies
RUN npm install

# Build application including libraries.
RUN npm run build

WORKDIR /app/api/dist

RUN npm install -g recursive-install
RUN npm-recursive-install --production

FROM node:16-alpine as production

WORKDIR /app/api

#COPY --from=builder /app/api/dist ./

EXPOSE 8080
CMD npm start
