FROM node:20-bookworm-slim

# Install tools needed for executing code/commands
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip bash curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Codex CLI for coding.delegate provider=codex
RUN npm install -g @openai/codex && codex --version

# Install OpenCode CLI for coding.delegate provider=opencode
RUN npm install -g opencode-ai@latest && opencode --version

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
