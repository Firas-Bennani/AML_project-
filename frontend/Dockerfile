FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . /app/

EXPOSE 5173

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=6 \
  CMD wget -qO- http://localhost:5173/ >/dev/null 2>&1 || exit 1

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
