import streamlit as st
import pandas as pd
import psycopg2
import requests
import os
import time
from datetime import datetime
import boto3
from botocore.config import Config

# --- Configuration ---
DB_CONFIG = {
    "host": "db",
    "port": 5432,
    "user": "nexus",
    "password": "nexus",
    "database": "nexus"
}
ORCH_URL = "http://orchestrator:3000"
APPROVAL_TOKEN = "dev-approval-token"

# MinIO Config
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "nexus")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "nexuspassword")
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000")
QUANT_LLM_MODEL = os.getenv("QUANT_LLM_MODEL", "deepseek-r1:32b")
CODE_LLM_MODEL = os.getenv("CODE_LLM_MODEL", "glm-4.7-flash:latest")

st.set_page_config(page_title="Nexus Commander Panel", layout="wide")

st.title("🤖 Nexus Commander Panel")
st.sidebar.markdown("### OpenClaw Nexus v1.2.2")
st.sidebar.info("Operational Control Plane for Autonomous Agents")

# --- Database Connection ---
def get_db_data(query, params=None):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return pd.DataFrame()

# --- MinIO Link Generation ---
def get_presigned_url(object_key):
    try:
        # We use a custom endpoint for the client (internal docker) 
        # but the presigned URL should ideally be accessible from host
        s3 = boto3.client('s3',
                          endpoint_url=MINIO_ENDPOINT,
                          aws_access_key_id=MINIO_ACCESS_KEY,
                          aws_secret_access_key=MINIO_SECRET_KEY,
                          config=Config(signature_version='s3v4'))
        
        url = s3.generate_presigned_url('get_object',
                                        Params={'Bucket': 'nexus-artifacts', 'Key': object_key},
                                        ExpiresIn=3600)
        
        # Replace internal container host with public localhost for browser access
        if "minio:9000" in url:
            url = url.replace("http://minio:9000", MINIO_PUBLIC_URL)
        return url
    except Exception as e:
        return f"Error: {e}"

# --- Approval Logic ---
def approve_task(task_id):
    try:
        res = requests.post(
            f"{ORCH_URL}/tasks/{task_id}/approve",
            headers={"X-Approval-Token": APPROVAL_TOKEN}
        )
        return res.status_code == 200
    except:
        return False

# --- UI Layout ---
tabs = st.tabs(["📊 Live Tasks", "🔐 Approval Hub", "📦 Asset Vault", "🚀 Quick Launch"])

# 1. Live Tasks
with tabs[0]:
    st.subheader("Latest System Activities")
    if st.button("Refresh Now"):
        st.rerun()
    
    tasks_df = get_db_data("SELECT task_id, tool_name, status, risk_level, created_at FROM tasks ORDER BY created_at DESC LIMIT 20")
    if not tasks_df.empty:
        def color_status(val):
            color = '#999'
            if val == 'succeeded': color = '#28a745'
            elif val == 'failed' or val == 'dlq': color = '#dc3545'
            elif val == 'running': color = '#007bff'
            elif val == 'waiting_approval': color = '#ffc107'
            return f'color: {color}; font-weight: bold'

        st.table(tasks_df.style.map(color_status, subset=['status']))
    else:
        st.write("No tasks found.")

# 2. Approval Hub
with tabs[1]:
    st.subheader("Tasks Requiring Review")
    pending_df = get_db_data("SELECT task_id, tool_name, risk_level, payload_json, created_at FROM tasks WHERE status = 'waiting_approval' ORDER BY created_at ASC")
    
    if not pending_df.empty:
        for _, row in pending_df.iterrows():
            task_time = row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            with st.expander(f"🔔 [{task_time}] {row['tool_name']} (Risk: {row['risk_level']})"):
                st.write(f"**Task ID:** `{row['task_id']}`")
                st.code(row['payload_json'], language="json")
                col1, col2 = st.columns(2)
                if col1.button("✅ Approve", key=f"app_{row['task_id']}"):
                    if approve_task(row['task_id']):
                        st.success("Approved!")
                        time.sleep(0.5)
                        st.rerun()
                if col2.button("❌ Reject", key=f"rej_{row['task_id']}"):
                    st.warning("Logic coming soon.")
    else:
        st.success("Clear! No pending approvals.")

# 3. Asset Vault
with tabs[2]:
    st.subheader("Archived Quant Artifacts (MinIO)")
    assets_df = get_db_data("""
        SELECT a.task_id, a.object_key, a.mime_type, a.created_at, t.tool_name 
        FROM assets a 
        JOIN tasks t ON a.task_id = t.task_id 
        ORDER BY a.created_at DESC
    """)
    if not assets_df.empty:
        for _, asset in assets_df.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"**{asset['object_key'].split('/')[-1]}** ({asset['tool_name']})")
            url = get_presigned_url(asset['object_key'])
            
            if col2.link_button("📂 Open", url):
                pass
            
            if asset['mime_type'] == 'text/html':
                if col3.button("👁? Preview", key=f"pre_{asset['task_id']}_{asset['object_key'].split('/')[-1]}"):
                    # Fetch content for preview
                    try:
                        s3 = boto3.client('s3',
                                          endpoint_url=MINIO_ENDPOINT,
                                          aws_access_key_id=MINIO_ACCESS_KEY,
                                          aws_secret_access_key=MINIO_SECRET_KEY)
                        obj = s3.get_object(Bucket='nexus-artifacts', Key=asset['object_key'])
                        html_content = obj['Body'].read().decode('utf-8')
                        st.session_state['preview_content'] = html_content
                        st.session_state['preview_title'] = asset['object_key']
                    except Exception as e:
                        st.error(f"Preview error: {e}")
            st.divider()
            
        if 'preview_content' in st.session_state:
            with st.sidebar:
                st.markdown(f"### 📄 Preview: {st.session_state['preview_title'].split('/')[-1]}")
                if st.button("Close Preview"):
                    del st.session_state['preview_content']
                    st.rerun()
                import streamlit.components.v1 as components
                components.html(st.session_state['preview_content'], height=800, scrolling=True)
    else:
        st.write("Vault is empty.")

# 4. Quick Launch
with tabs[3]:
    st.subheader("Launch Quant Pipeline")
    symbol = st.text_input("Stock Ticker", "NVDA")
    model_choice = st.selectbox(
        "Analysis Model",
        options=[QUANT_LLM_MODEL, CODE_LLM_MODEL, "deepseek-r1:32b", "glm-4.7-flash:latest"],
        index=0,
    )
    # Normalize "ollama/..." to raw model id for direct Ollama calls
    if isinstance(model_choice, str) and model_choice.startswith("ollama/"):
        model_choice = model_choice.split("/", 1)[1]
    if st.button("🚀 Run AI-Powered Pipeline"):
        wf_payload = {
            "name": f"UI Launch: {symbol}",
            "definition": {
                "steps": [
                    {"tool_name": "quant.fetch_price", "payload": {"symbol": symbol}, "risk_level": "low"},
                    {"tool_name": "quant.run_optimized_pipeline", "payload": {}, "risk_level": "medium"},
                    {"tool_name": "ai.analyze", "payload": {"model": model_choice}, "risk_level": "low"},
                    {"tool_name": "media.generate_report_card", "payload": {}, "risk_level": "low"}
                ]
            }
        }
        res = requests.post(f"{ORCH_URL}/workflows", json=wf_payload)
        if res.status_code == 200:
            st.success("Pipeline Launched!")
        else:
            st.error("Launch failed.")

    st.divider()
    st.subheader("Launch Daily Reports")
    if st.button("?? Run Market News Report"):
        wf_payload = {
            "name": "Daily Market News",
            "definition": {"steps": [{"tool_name": "news.daily_report", "payload": {}, "risk_level": "low"}]},
        }
        res = requests.post(f"{ORCH_URL}/workflows", json=wf_payload)
        if res.status_code == 200:
            st.success("News report started!")
        else:
            st.error("Launch failed.")

    if st.button("?? Run GitHub Skills Report"):
        wf_payload = {
            "name": "Daily GitHub Skills",
            "definition": {"steps": [{"tool_name": "github.skills_daily_report", "payload": {}, "risk_level": "low"}]},
        }
        res = requests.post(f"{ORCH_URL}/workflows", json=wf_payload)
        if res.status_code == 200:
            st.success("GitHub report started!")
        else:
            st.error("Launch failed.")

# Auto-refresh
time.sleep(5)
st.rerun()
