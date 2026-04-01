{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://host.docker.internal:11434/v1"
      },
      "models": {
        "glm-4.7-flash:latest": {
          "name": "glm-4.7-flash:latest"
        },
        "deepseek-r1:32b": {
          "name": "deepseek-r1:32b"
        },
        "qwq:latest": {
          "name": "qwq:latest"
        }
      }
    },
    "dashscope": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DashScope (Cloud)",
      "options": {
        "baseURL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "apiKey": "${DASH_SCOPE_API_KEY}"
      },
      "models": {
        "qwen-flash-2025-07-28": {
          "name": "qwen-flash-2025-07-28"
        },
        "qwen-plus-2025-04-28": {
          "name": "qwen-plus-2025-04-28",
          "options": {
            "maxTokens": 16000
          }
        }
      }
    },
    "minimax-coding-plan": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MiniMax (minimaxi.com)",
      "options": {
        "baseURL": "https://api.minimaxi.com/v1",
        "apiKey": "${MINIMAX_API_KEY}"
      },
      "models": {
        "MiniMax-M2.7": {
          "name": "MiniMax-M2.7"
        },
        "MiniMax-M2.5": {
          "name": "MiniMax-M2.5"
        }
      }
    },
    "dashscope-coder": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DashScope Coder",
      "options": {
        "baseURL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "apiKey": "${DASH_SCOPE_API_KEY}"
      },
      "models": {
        "qwen-plus-2025-04-28": {
          "name": "qwen-plus-2025-04-28",
          "options": {
            "maxTokens": 16000
          }
        },
        "qwen3-coder-plus": {
          "name": "qwen3-coder-plus"
        }
      }
    }
  }
}
