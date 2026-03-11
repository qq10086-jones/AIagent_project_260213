# News Worker — 自动化新闻收集与分析工具
#
# 模块：
#   news_ingester.py   抓取新闻，写入 news_feed 表
#   news_analyzer.py   事件聚类 + LLM 打分，写入 news_sentiment 表
#
# 设计文档：news/design/
#   NEWS_SYSTEM_DESIGN_v1.md   架构设计
#   NEWS_GOVERNANCE_v1.md      治理规范
#   NEWS_SYSTEM_TASKS_v1.md    任务清单
