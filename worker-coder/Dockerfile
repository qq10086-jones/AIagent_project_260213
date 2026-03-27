FROM node:20-bookworm-slim

# Install tools needed for executing code/commands
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip bash curl ca-certificates gettext-base \
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

# Copy and register entrypoint (resolves ${VAR} in opencode.json.tpl at startup)
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Start worker via entrypoint so env vars are substituted into opencode.json
CMD ["/bin/sh", "entrypoint.sh"]
