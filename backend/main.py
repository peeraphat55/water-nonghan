# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
import pandas as pd
import os
import pickle
# pyrefly: ignore [missing-import]
from groq import Groq
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# โหลดตัวแปร environment (.env ที่โฟลเดอร์แม่)
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(os.path.dirname(base_dir), ".env")
load_dotenv(env_path)

# เริ่มต้น Groq Client แบบ Async (เหมาะกับ FastAPI)
from groq import AsyncGroq
client = AsyncGroq(
    api_key=os.environ.get("API_GROQ_KEY"),
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# โหลดข้อมูล CSV
csv_path = os.path.join(base_dir, "dataset1_nonghan_water_quality.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "FastAPI is running successfully!"}

@app.post("/predict-result")
def predict_result():
    return "nothing"

@app.post("/summary-in-past")
def summary_in_past():
    return {"message": "Summary in past"}

@app.post("/chat")
async def chat_with_data(request: ChatRequest):
    if df.empty:
        return {"reply": "ไม่พบไฟล์ข้อมูล หรือข้อมูลว่างเปล่า"}
        
    # เตรียม Context เพื่อส่งให้ LLM
    stations = ", ".join(df['station_id'].dropna().unique())
    tambons = ", ".join(df['tambon'].dropna().unique())
    wqi_counts = ", ".join([f"{k}: {v} รายการ" for k, v in df['WQI_al_class'].value_counts().items()])
    seasons = ", ".join([f"{k}: {v} รายการ" for k, v in df['season'].value_counts().items()])
    
    # จัดอันดับสถานีตามคะแนน WQI เฉลี่ย พร้อมแสดงข้อมูลตำบลและประเภท
    station_info = df.groupby('station_id').first()[['tambon', 'station_type']]
    station_wqi = df.groupby('station_id')['WQI_al_score'].mean().sort_values()
    
    station_ranking_list = []
    for station, score in station_wqi.items():
        tambon = station_info.loc[station, 'tambon']
        stype = station_info.loc[station, 'station_type']
        station_ranking_list.append(f"  - สถานี {station} (ตำบล{tambon}, {stype}): {score:.2f} คะแนน")
    
    station_ranking = "\n".join(station_ranking_list)
    # เปรียบเทียบตามประเภทแหล่งน้ำ (station_type)
    type_wqi = df.groupby('station_type')['WQI_al_score'].mean() if 'station_type' in df.columns else {}
    type_comparison = ", ".join([f"{stype}: {score:.2f} คะแนน" for stype, score in type_wqi.items()])

    context = f"""
ข้อมูลภาพรวมคุณภาพน้ำหนองหารและลำห้วยที่คุณมีคือ:
- จำนวนข้อมูลทั้งหมด: {len(df)} รายการ
- สถานีเก็บตัวอย่างทั้งหมด: {stations}
- ตำบลที่มีการเก็บข้อมูล: {tambons}
- สัดส่วนข้อมูลตามฤดูกาล: {seasons}
- ค่า pH: เฉลี่ย {df['pH'].mean():.2f}, ต่ำสุด {df['pH'].min():.2f}, สูงสุด {df['pH'].max():.2f}
- ค่า DO (ออกซิเจนละลายน้ำ): เฉลี่ย {df['DO_mg_L'].mean():.2f} mg/L
- คุณภาพน้ำตามเกณฑ์ WQI (จำนวนรายการ): {wqi_counts}
- เปรียบเทียบคะแนน WQI เฉลี่ยตามประเภทแหล่งน้ำ (ลำห้วย vs หนองหาร): {type_comparison}
- ข้อมูลคะแนน WQI เฉลี่ยรายสถานี เรียงจากน้อยไปมาก (คะแนนน้อย = แย่ที่สุด, คะแนนมาก = ดีที่สุด):
{station_ranking}
"""

    system_prompt = (
        "คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญด้านข้อมูลคุณภาพน้ำหนองหาร "
        "จงตอบคำถามโดยอิงจากข้อมูลต่อไปนี้เท่านั้น "
        "หากคำถามอยู่นอกเหนือจากข้อมูลนี้ให้บอกอย่างสุภาพว่าไม่มีข้อมูล "
        "และกรุณาตอบคำถามอย่างเป็นธรรมชาติและเป็นมิตรด้วยภาษาไทย\n\n"
        f"{context}"
    )

    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            model="llama-3.3-70b-versatile",
        )
        return {"reply": chat_completion.choices[0].message.content}
    except Exception as e:
        return {"reply": f"เกิดข้อผิดพลาดในการเรียกใช้งาน Groq API: {str(e)}"}