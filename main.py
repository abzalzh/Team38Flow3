import asyncio
from contextlib import asynccontextmanager
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional, List
import uuid
import os
import json
import base64
import urllib.parse
from pydantic import BaseModel
import traceback
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import platform

# --- NEW IMPORTS FOR CAMERA ---
import cv2
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
PORT = 8000
USERS_DB_FILE = "users.json"
SQLITE_FILE = "water_registry.db"
DATABASE_URL = f"sqlite:///{SQLITE_FILE}"

# --- TELEGRAM CONFIG ---
TELEGRAM_BOT_TOKEN = "8270647703:AAErIAhAf7PlDe6kJdgdrZ7dqqB_ZFKaD8I" 

# --- AI CAMERA CONFIG ---
# Using the path you provided. If this file is missing, it will fallback to standard YOLO.
MODEL_PATH = "abzal/temp/runs/detect/train2/weights/best.pt" 
CAM_ID = "cam1"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{CAM_ID}_log.txt")

# Global variables for Camera
camera_running = False
connected_clients = [] # List of WebSockets watching the stream
ai_model = None

# --- TELEGRAM DB MODEL ---
class TelegramConnection(SQLModel, table=True):
    email: str = Field(primary_key=True)
    telegram_id: int
    username: Optional[str] = None
    connected_at: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Run DB init, load AI model, and start Bot
    init_db()
    load_user_db()
    
    # Load AI Model
    global ai_model
    try:
        print(f"🔄 Loading YOLO model from: {MODEL_PATH}")
        ai_model = YOLO(MODEL_PATH)
        print("✅ YOLO Model Loaded Successfully")
    except Exception as e:
        print(f"⚠️ Warning: Custom model not found ({e}). Downloading standard yolov8n.pt...")
        ai_model = YOLO("yolov8n.pt")

    asyncio.create_task(dp.start_polling(bot))
    print("--- TELEGRAM BOT STARTED ---")
    yield
    # Shutdown: Close bot session
    await bot.session.close()

app = FastAPI(lifespan=lifespan)

# ==========================================
# 2. AI CAMERA LOGIC (MERGED)
# ==========================================

def detect_people(frame):
    """
    Runs YOLO detection on the frame.
    Returns: frame with boxes, count of people.
    """
    if ai_model is None: 
        return frame, 0
    
    # Run inference (classes=[0] is usually Person in COCO, adjust if your custom model differs)
    results = ai_model(frame, classes=[0], verbose=False) 

    count = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Person {conf:.2f}"
            
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            count += 1
    return frame, count

async def camera_stream_task():
    """
    Background task that opens the camera, processes frames, and sends them to the frontend.
    """
    global camera_running
    print("📷 Camera: ON (Starting Stream)")
    
    # Open Camera (Index 0 is default webcam)
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera (Index 0).")
        camera_running = False
        return

    while camera_running and len(connected_clients) > 0:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue

        # 1. AI Processing
        frame, count = detect_people(frame)

        # 2. Logging (Write to file)
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE, "a") as f:
                f.write(f"{timestamp} | {count}\n")
        except Exception as e:
            print(f"Log Error: {e}")

        # 3. Encode to JPEG -> Base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # 4. Broadcast to all connected browsers
        disconnect_list = []
        for ws in connected_clients:
            try:
                await ws.send_text(jpg_as_text)
            except:
                disconnect_list.append(ws)
        
        # Cleanup disconnected clients
        for ws in disconnect_list:
            if ws in connected_clients:
                connected_clients.remove(ws)
        
        # Limit FPS (~30 FPS)
        await asyncio.sleep(0.03) 

    cap.release()
    print("📷 Camera: OFF (No clients or stopped manually)")
    camera_running = False

@app.websocket("/ws/live_camera")
async def websocket_camera_endpoint(websocket: WebSocket):
    """
    WebSocket Endpoint for the website to connect to.
    """
    global camera_running
    
    await websocket.accept()
    connected_clients.append(websocket)
    print("✅ Client connected to Video Stream")
    
    # If camera isn't running, start the background task
    if not camera_running:
        camera_running = True
        asyncio.create_task(camera_stream_task())
    
    try:
        while True:
            # Keep connection open, wait for messages (optional)
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("❌ Client disconnected from Video Stream")
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        # The camera stream task checks 'connected_clients' length and will stop itself if empty

# ==========================================
# 3. SECURITY & AUTHENTICATION (JSON)
# ==========================================

def encrypt_pass(text):
    return base64.b64encode(urllib.parse.quote(text).encode('utf-8')).decode('utf-8')

# Default Users
DEFAULT_USERS = [
    {"name": "Admin", "surname": "Expert", "email": "admin@hydro.com", "pass": encrypt_pass("admin123"), "status": "Expert"},
    {"name": "Guest", "surname": "User", "email": "guest@hydro.com", "pass": encrypt_pass("guest123"), "status": "Guest"}
]

def load_user_db():
    if not os.path.exists(USERS_DB_FILE):
        save_user_db(DEFAULT_USERS)
        return DEFAULT_USERS
    try:
        with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading user DB: {e}")
        return []

def save_user_db(data):
    with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Pydantic Models
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    name: str
    surname: str
    email: str
    password: str

class RoleRequest(BaseModel):
    email: str

# ==========================================
# 4. WATER OBJECTS DATABASE (SQL)
# ==========================================

class WaterObject(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    region: str
    type: str
    status: int
    coordinates_lat: float
    coordinates_lng: float
    passportDate: str
    area: Optional[str] = "Н/Д"
    capacity: Optional[str] = "Н/Д"
    resource_type: Optional[str] = "other"
    technical_condition: Optional[int] = 0
    pdf_url: Optional[str] = "-"
    added_by: Optional[str] = "System"

engine = create_engine(DATABASE_URL)

def init_db():
    SQLModel.metadata.create_all(engine)

class WaterObjectCreate(BaseModel):
    name: str
    region: str
    type: str = "other"
    resource_type: Optional[str] = "other"
    status: int = 0
    technical_condition: Optional[int] = 0
    coordinates_lat: float
    coordinates_lng: float
    passportDate: str = ""
    passport_date: Optional[str] = ""
    area: Optional[str] = ""
    capacity: Optional[str] = ""
    pdf_url: Optional[str] = ""
    added_by: Optional[str] = "Expert"

# ==========================================
# 5. TELEGRAM BOT LOGIC
# ==========================================
class ConnectionStates(StatesGroup):
    waiting_for_email = State()
    waiting_for_password = State()

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

def verify_user_credential(email, password):
    users = load_user_db()
    enc_pass = encrypt_pass(password)
    for user in users:
        if user['email'].lower() == email.lower():
            if user['pass'] == enc_pass:
                return user
    return None

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    with Session(engine) as session:
        conn = session.exec(select(TelegramConnection).where(TelegramConnection.telegram_id == message.from_user.id)).first()
    
    if conn:
        await message.answer(f"✅ Вы уже подключены как: {conn.email}")
    else:
        keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
            [types.InlineKeyboardButton(text="🔗 Подключить аккаунт", callback_data="connect")]
        ])
        await message.answer("🤖 <b>HydroAtlas Monitor</b>\nНажмите кнопку ниже, чтобы подключить ваш аккаунт и получать уведомления.", reply_markup=keyboard, parse_mode="HTML")

@dp.callback_query(F.data == "connect")
async def process_connect(callback: types.CallbackQuery, state: FSMContext):
    await state.set_state(ConnectionStates.waiting_for_email)
    await callback.message.answer("📧 Введите ваш Email от HydroAtlas:")

@dp.message(ConnectionStates.waiting_for_email)
async def process_email(message: types.Message, state: FSMContext):
    await state.update_data(email=message.text.strip())
    await state.set_state(ConnectionStates.waiting_for_password)
    await message.answer("🔑 Введите ваш пароль:")

@dp.message(ConnectionStates.waiting_for_password)
async def process_password(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    email = user_data.get('email')
    password = message.text.strip()
    
    user = verify_user_credential(email, password)
    
    if user:
        if user['status'] != 'Expert':
            await message.answer("❌ Только пользователи со статусом <b>Expert</b> могут подключать бота.", parse_mode="HTML")
        else:
            try:
                with Session(engine) as session:
                    existing = session.get(TelegramConnection, email)
                    if existing: session.delete(existing)
                    
                    new_conn = TelegramConnection(
                        email=email,
                        telegram_id=message.from_user.id,
                        username=message.from_user.username,
                        connected_at=str(asyncio.get_event_loop().time())
                    )
                    session.add(new_conn)
                    session.commit()
                await message.answer(f"✅ <b>Успешно!</b>\nАккаунт {email} подключен.\nТеперь вы можете получать уведомления с сайта.", parse_mode="HTML")
            except Exception as e:
                await message.answer(f"Ошибка базы данных: {e}")
    else:
        await message.answer("❌ Неверный логин или пароль. Введите /start чтобы попробовать снова.")
    await state.clear()

# ==========================================
# 6. API ENDPOINTS
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Error: index.html not found in server directory.</h1>")

@app.get("/api/users")
def get_users():
    return load_user_db()

@app.post("/api/login")
def login(req: LoginRequest):
    users = load_user_db()
    enc_pass = encrypt_pass(req.password)
    user = next((u for u in users if u['email'] == req.email and u['pass'] == enc_pass), None)
    
    if user:
        token = f"TOKEN_{base64.b64encode(req.email.encode()).decode()}_{uuid.uuid4()}"
        return {"status": "ok", "token": token, "role": user['status'], "user": user}
    else:
        return JSONResponse(status_code=401, content={"error": "Неверный логин или пароль"})

@app.post("/api/register")
def register(req: RegisterRequest):
    users = load_user_db()
    if any(u['email'] == req.email for u in users):
        return JSONResponse(status_code=409, content={"error": "Email уже зарегистрирован"})
    
    new_user = {
        "name": req.name, "surname": req.surname, "email": req.email,
        "pass": encrypt_pass(req.password), "status": "Guest"
    }
    users.append(new_user)
    save_user_db(users)
    token = f"TOKEN_{base64.b64encode(req.email.encode()).decode()}_{uuid.uuid4()}"
    return {"status": "ok", "token": token, "role": "Guest", "user": new_user}

@app.post("/api/update_role")
def update_role(req: RoleRequest):
    users = load_user_db()
    for u in users:
        if u['email'] == req.email:
            u['status'] = 'Expert' if u['status'] == 'Guest' else 'Guest'
            save_user_db(users)
            return {"status": "ok", "new_role": u['status']}
    return JSONResponse(status_code=404, content={"error": "User not found"})

@app.get("/api/water")
def get_water_objects():
    with Session(engine) as session:
        objs = session.exec(select(WaterObject)).all()
        result = []
        for o in objs:
            result.append({
                "id": 10000 + (o.id or 0), 
                "name": o.name,
                "region": o.region,
                "type": o.type,
                "resource_type": o.resource_type,
                "status": o.status,
                "technical_condition": o.technical_condition,
                "coordinates": [o.coordinates_lat, o.coordinates_lng],
                "latitude": o.coordinates_lat,
                "longitude": o.coordinates_lng,
                "passportDate": o.passportDate,
                "area": o.area,
                "capacity": o.capacity,
                "pdf_url": o.pdf_url,
                "is_custom": True
            })
        return result

@app.post("/api/water")
def add_water_object(obj: WaterObjectCreate):
    try:
        res_type = obj.resource_type if obj.resource_type else obj.type
        tech_cond = obj.technical_condition if obj.technical_condition else obj.status
        p_date = obj.passport_date if obj.passport_date else obj.passportDate

        db_obj = WaterObject(
            name=obj.name, region=obj.region, type=obj.type, resource_type=res_type,
            status=obj.status, technical_condition=tech_cond,
            coordinates_lat=obj.coordinates_lat, coordinates_lng=obj.coordinates_lng,
            passportDate=p_date, area=obj.area, capacity=obj.capacity,
            pdf_url=obj.pdf_url, added_by=obj.added_by
        )
        with Session(engine) as session:
            session.add(db_obj)
            session.commit()
            session.refresh(db_obj)
        return {"status": "ok", "id": db_obj.id, "message": "Object added successfully"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- PDF Generation ---
def get_cyrillic_font():
    font_name = "Helvetica"
    possible_paths = ["arial.ttf", "C:\\Windows\\Fonts\\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('RusFont', path))
                return 'RusFont'
            except: continue
    return font_name

@app.post("/api/generate_pdf")
async def generate_pdf(obj: dict):
    filename = "Regions Database.pdf"
    filepath = os.path.join(os.getcwd(), filename)
    try:
        c = canvas.Canvas(filepath, pagesize=A4)
        width, height = A4
        font_name = get_cyrillic_font()
        c.setFont(font_name, 18)
        c.drawString(50, height - 50, "HYDROATLAS: PASSPORT OF OBJECT")
        c.setFont(font_name, 10)
        c.drawString(50, height - 70, f"Generated automatically by System | Date: {obj.get('passportDate', 'N/A')}")
        c.line(50, height - 80, width - 50, height - 80)
        y = height - 120
        c.setFont(font_name, 14)
        c.drawString(50, y, f"OBJECT NAME: {obj.get('name', 'Unknown')}")
        y -= 30
        c.setFont(font_name, 12)
        info_map = [("Region", obj.get('region', '-')), ("Type", obj.get('type', '-')), ("Technical Status", f"{obj.get('status', '-')}/5"), ("Coordinates", f"{obj.get('coordinates', '-')}")]
        if obj.get('area'): info_map.append(("Area", obj.get('area')))
        if obj.get('capacity'): info_map.append(("Capacity", obj.get('capacity')))
        if obj.get('priorityCat'): info_map.append(("Priority", obj.get('priorityCat', {}).get('text', '-')))
        if obj.get('mlProb'): info_map.append(("ML Risk Probability", f"{obj.get('mlProb')}%"))
        
        for label, value in info_map:
            c.drawString(50, y, f"{label}: {str(value)}")
            y -= 25
        c.save()
        return FileResponse(path=filepath, filename=filename, media_type='application/pdf')
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Telegram API ---
class TelegramReportRequest(BaseModel):
    user_email: str
    objects: List[dict]

class TestNotifRequest(BaseModel):
    user_email: str

@app.post("/api/telegram/status")
def telegram_status(body: dict):
    email = body.get('email')
    with Session(engine) as session:
        conn = session.get(TelegramConnection, email)
        if conn: return {"connected": True, "telegram_id": conn.telegram_id}
    return {"connected": False}

@app.post("/api/telegram/test")
async def send_test_notification(req: TestNotifRequest):
    with Session(engine) as session:
        conn = session.get(TelegramConnection, req.user_email)
        if not conn: return JSONResponse(status_code=404, content={"error": "Telegram not connected"})
        try:
            await bot.send_message(conn.telegram_id, "🔔 <b>TEST NOTIFICATION</b>\nSystem is working!", parse_mode="HTML")
            return {"status": "ok"}
        except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/telegram/report")
async def send_report(req: TelegramReportRequest):
    with Session(engine) as session:
        conn = session.get(TelegramConnection, req.user_email)
        if not conn: return JSONResponse(status_code=404, content={"error": "Telegram not connected"})
        msg = "🚨 <b>CRITICAL OBJECTS REPORT</b>\n\n"
        for i, obj in enumerate(req.objects):
            msg += f"{i+1}. <b>{obj.get('name')}</b> ({obj.get('region')})\n   ⚠️ Status: {obj.get('status')}/5\n\n"
        try:
            await bot.send_message(conn.telegram_id, msg, parse_mode="HTML")
            return {"status": "ok"}
        except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    print("-------------------------------------------------------")
    print("STARTING HYDROATLAS SERVER")
    print("-------------------------------------------------------")
    init_db()
    load_user_db()
    print(f"Server is running at: http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)