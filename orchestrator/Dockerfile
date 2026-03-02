FROM node:20-alpine
WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev
COPY src ./src
COPY configs ./configs
EXPOSE 3000
CMD ["npm","start"]
