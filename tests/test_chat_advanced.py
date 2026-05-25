import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import io
import uuid
import logging
from unittest.mock import patch
from fastapi.testclient import TestClient

from nava_core.gathi.api.main import app
from nava_core.mozhi.chat.client import ChatClient
from nava_core.mozhi.chat.service import ChatService

# To capture prompt
PROMPT_LOG = []

original_send = ChatClient.send
def spy_send(self, messages, *args, **kwargs):
    PROMPT_LOG.append(messages)
    return original_send(self, messages, *args, **kwargs)

# We also want to capture the generated summary. The summary is generated in ChatService._summarize_if_needed.
# We can spy on _extract_crop_notes_from_summary to see what summary it received.
SUMMARY_LOG = []
original_extract = ChatService._extract_crop_notes_from_summary
def spy_extract(self, summary_text: str, crop_id: int):
    SUMMARY_LOG.append(summary_text)
    return original_extract(self, summary_text, crop_id)


def setup_field_crop_plant(client, token):
    headers = {"Authorization": f"Bearer {token}"}
    res = client.get("/api/fields", headers=headers)
    fields = res.json().get("fields", [])
    field_id = None
    for f in fields:
        if f["name"] == "Advanced Test Field":
            field_id = f["id"]
            break
            
    if not field_id:
        res = client.post("/api/fields", headers=headers, json={
            "name": "Advanced Test Field", "location": "Test Farm", "area": "1 acre", "soil_type": "Loamy"
        })
        field_id = res.json()["id"]
    
    res = client.get(f"/api/crops?field_id={field_id}", headers=headers)
    crops = res.json().get("crops", [])
    crop_id = None
    for c in crops:
        if c["name"] == "Banana":
            crop_id = c["id"]
            break
            
    if not crop_id:
        res = client.post("/api/crops", headers=headers, json={
            "field_id": field_id, "name": "Banana", "variety": "Cavendish", "season": "Summer", "stage": "Vegetative"
        })
        crop_id = res.json()["id"]
    return field_id, crop_id

def test_chat_advanced():
    print("--- Starting Advanced Chat Tests ---")
    os.makedirs("tests/outputs", exist_ok=True)
    report_path = "tests/chat_report.md"
    
    # Setup custom logger to capture routing and RAG logs
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    nava_logger = logging.getLogger("nava")
    nava_logger.addHandler(ch)
    # Ensure nava_logger is capturing debug if needed
    nava_logger.setLevel(logging.DEBUG)
    
    with TestClient(app) as client:
        with patch.object(ChatClient, 'send', new=spy_send), \
             patch.object(ChatService, '_extract_crop_notes_from_summary', new=spy_extract):
            
            # Register or Login fixed user
            res = client.post("/api/auth/login", json={
                "email": "test@gmail.com", "password": "123456789"
            })
            if res.status_code == 200:
                token = res.json()["token"]
            else:
                res = client.post("/api/auth/register", json={
                    "name": "Test Farmer", "email": "test@gmail.com", "password": "123456789"
                })
                token = res.json()["token"]
            
            headers = {"Authorization": f"Bearer {token}"}
            field_id, crop_id = setup_field_crop_plant(client, token)
            session_id = f"test_chat_session"
            
            def chat_msg(msg):
                import time
                time.sleep(3)
                
                PROMPT_LOG.clear()
                # clear string buffer
                log_capture_string.seek(0)
                log_capture_string.truncate(0)
                
                res = client.post("/api/chat", headers=headers, json={
                    "message": msg, "session_id": session_id, "field_id": field_id, "crop_id": crop_id
                })
                
                logs = log_capture_string.getvalue().replace('\\n', '\n')
                prompt = PROMPT_LOG[-1] if PROMPT_LOG else None
                if prompt:
                    import json
                    prompt = json.dumps(prompt, indent=2).replace('\\n', '\n')
                return res.json(), logs, prompt
            
            with open(report_path, "w") as md:
                md.write("# Advanced Chat & RAG Qualitative Report\n\n")
                
                # Step 1
                msg = "Hello NAVA, how are you today?"
                print(f"Test 1: {msg}")
                data, logs, prompt = chat_msg(msg)
                md.write("## 1. General Greeting\n")
                md.write(f"**User**: {msg}\n\n")
                md.write(f"**NAVA**: {data.get('reply')}\n\n")
                md.write(f"**RAG Used**: {data.get('rag_used')}\n\n")
                md.write("### Routing Logs\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-size: 12px;'>\n" + logs.strip() + "\n</pre>\n\n")
                
                # Step 2
                msg = "Can you give me a detailed overview of my field?"
                print(f"Test 2: {msg}")
                data, logs, prompt = chat_msg(msg)
                md.write("## 2. Context Question\n")
                md.write(f"**User**: {msg}\n\n")
                md.write(f"**NAVA**: {data.get('reply')}\n\n")
                md.write(f"**RAG Used**: {data.get('rag_used')}\n\n")
                md.write("### Routing Logs\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-size: 12px;'>\n" + logs.strip() + "\n</pre>\n\n")
                if prompt:
                    md.write("### LLM Prompt Sent\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-size: 12px;'>\n" + str(prompt) + "\n</pre>\n\n")
                    
                # Step 3
                msg = "I suspect my banana plants have Black Sigatoka. What is the recommended treatment?"
                print(f"Test 3: {msg}")
                data, logs, prompt = chat_msg(msg)
                md.write("## 3. RAG Question\n")
                md.write(f"**User**: {msg}\n\n")
                md.write(f"**NAVA**: {data.get('reply')}\n\n")
                md.write(f"**RAG Used**: {data.get('rag_used')} (Chunks: {data.get('rag_chunk_count')})\n\n")
                md.write("### RAG Retrieval & Routing Logs\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-size: 12px;'>\n" + logs.strip() + "\n</pre>\n\n")
                if prompt:
                    md.write("### LLM Prompt Sent\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-size: 12px;'>\n" + str(prompt) + "\n</pre>\n\n")
                    
                # Step 4
                msg = "Okay, that is enough for me, thank you!"
                print(f"Test 4: {msg}")
                data, logs, prompt = chat_msg(msg)
                md.write("## 4. Basic Follow-up\n")
                md.write(f"**User**: {msg}\n\n")
                md.write(f"**NAVA**: {data.get('reply')}\n\n")
                md.write(f"**RAG Used**: {data.get('rag_used')}\n\n")
                md.write("### Routing Logs\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-size: 12px;'>\n" + logs.strip() + "\n</pre>\n\n")
                
                # Step 5
                msg = "I have applied Mancozeb fungicide today."
                print(f"Test 5: {msg}")
                data, logs, prompt = chat_msg(msg)
                md.write("## 5. Agronomic Action Statement\n")
                md.write(f"**User**: {msg}\n\n")
                md.write(f"**NAVA**: {data.get('reply')}\n\n")
                
                # Step 6 - dummy messages
                dummies = ["That's all thank you", "Bye bye"]
                md.write("## 6. Dummy Messages (Triggering Summarizer)\n")
                for dm in dummies:
                    print(f"Dummy: {dm}")
                    data, logs, prompt = chat_msg(dm)
                    md.write(f"- **User**: {dm} -> **NAVA**: {data.get('reply')}\n")
                md.write("\n")
                
                # Now fetch the crop to see the auto notes
                res = client.get(f"/api/crops?field_id={field_id}", headers=headers)
                crops = res.json().get("crops", [])
                notes = crops[0].get("notes", "") if crops else ""
                
                summary_text = SUMMARY_LOG[-1] if SUMMARY_LOG else "No summary captured"
                
                md.write("## 7. Generated Summary & Auto-Notes\n\n")
                md.write("### Raw Chat Summary\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-size: 12px;'>\n" + summary_text + "\n</pre>\n\n")
                md.write("### Extracted Crop Notes\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-size: 12px;'>\n" + notes + "\n</pre>\n\n")
                
    nava_logger.removeHandler(ch)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    test_chat_advanced()
