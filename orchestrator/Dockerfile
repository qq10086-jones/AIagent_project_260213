FROM node:20-alpine
WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev
COPY src ./src
COPY configs ./configs
COPY contracts ./contracts
RUN rm -rf ./configs/llm_providers.json ./configs/llm_role_policy.json ./configs/context_budget_policy.json \
    && touch ./configs/llm_providers.json ./configs/llm_role_policy.json ./configs/context_budget_policy.json
EXPOSE 3000
CMD ["npm","start"]
