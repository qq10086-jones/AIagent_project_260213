FROM node:20-alpine

# Install tools needed for executing code/commands
RUN apk add --no-cache git python3 py3-pip bash curl

# Install Codex CLI for coding.delegate provider=codex
RUN npm install -g @openai/codex && codex --version

WORKDIR /app

# Copy package files
COPY package.json ./
RUN npm install --production

# Copy source files
COPY . .

# Environment variables
ENV NODE_ENV=production

# Start worker
CMD ["node", "worker.js"]
