# Builds the Vite/React SPA and serves it from nginx, proxying /api to the rusty-gpt server.
# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY tsconfig*.json vite.config.ts index.html eslint.config.js ./
COPY public ./public
COPY src ./src

RUN npm run build


FROM nginx:1.27-alpine AS runtime
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
