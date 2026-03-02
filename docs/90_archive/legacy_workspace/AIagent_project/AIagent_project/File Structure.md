/My_AI_Nexus
│
├── /core (OpenClaw 主程序)
│   ├── config.json (LLM配置)
│   └── SOUL.md (中枢人设)
│
├── /skills (自定义触角脚本)
│   ├── /quant_trade (量化相关)
│   │   ├── sbi_login.py
│   │   └── strategy_executor.py
│   │
│   ├── /ecommerce_ops (电商相关)
│   │   ├── mercari_lister.ts
│   │   └── qoo10_chat.py
│   │
│   └── /content_creation (内容生产)
│       ├── novel_generator.py
│       └── comfyui_api_client.py
│
├── /memory (长期记忆)
│   ├── trade_logs.md
│   └── product_inventory.db
│
└── /security (安全审计)
    └── pii_filter.py