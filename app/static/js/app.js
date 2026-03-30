// app/static/js/app.js

const API_BASE_URL = '/api';

document.addEventListener('DOMContentLoaded', () => {
    loadFields();
});

// ==========================================
// 1. ระบบจัดการหน้า (Navigation SPA)
// ==========================================
function switchTab(tabName) {
    // รีเซ็ตปุ่ม Tab กลับเป็นสีลางๆ
    document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
    // รีเซ็ตหน้าเนื้อหา ซ่อนให้หมด
    document.getElementById('content-fields').classList.add('d-none');
    document.getElementById('content-predict').classList.add('d-none');
    document.getElementById('content-history').classList.add('d-none');

    // เปิดของใหม่ที่ถูกเลือก
    document.getElementById(`tab-${tabName}`).classList.add('active');
    document.getElementById(`content-${tabName}`).classList.remove('d-none');

    // ถ้ากดเข้าหน้า History ให้เช็คเผื่อมีแปลงเลือกค้างไว้แล้วให้โหลด
    if (tabName === 'history') {
        const hId = document.getElementById('fieldSelectHistory').value;
        if (hId) loadHistory(hId);
    }
}

// ฟังก์ชันทางลัดสำหรับกระโดดไปหน้าตรวจความพร้อมของแปลงนี้
function jumpToPredict(fieldId) {
    switchTab('predict');
    document.getElementById('fieldSelectPredict').value = fieldId;

    // รีเซ็ตแบบฟอร์มเดิมเผื่อมีขยะ
    document.getElementById('soilMoisture').value = '';
    document.getElementById('nitrogenLevel').value = '';
    document.getElementById('resultArea').classList.add('d-none');
}

// ==========================================
// 2. ระบบจัดการแปลง (Field Management)
// ==========================================
async function loadFields() {
    try {
        const response = await fetch(`${API_BASE_URL}/fields/`);
        const fields = await response.json();

        const grid = document.getElementById('fieldsGrid');
        const selPredict = document.getElementById('fieldSelectPredict');
        const selHistory = document.getElementById('fieldSelectHistory');

        grid.innerHTML = '';
        selPredict.innerHTML = '<option value="" disabled selected>-- กรุณาเลือกแปลง --</option>';
        selHistory.innerHTML = '<option value="" disabled selected>-- กรุณาเลือกแปลง --</option>';

        if (fields.length === 0) {
            grid.innerHTML = '<div class="col-12 text-center text-muted py-4">ย้งไม่มีแปลงปลูก กดสร้างด้านบนได้เลย!</div>';
            return;
        }

        const plantNameMap = {
            "Spinach": "ปวยเล้ง (Spinach)",
            "Lettuce": "ผักกาดหอม (Lettuce)",
            "Kale": "ผักเคล (Kale)",
            "Swiss Chard": "สวิสชาร์ด (Swiss Chard)",
            "Bok Choy": "กวางตุ้งฮ่องเต้ (Bok Choy)",
            "Watercress": "สลัดน้ำ (Watercress)",
            "Arugula": "ผักร็อกเก็ต (Arugula)",
            "Mustard Greens": "ผักกาดเขียว (Mustard Greens)",
            "Collard Greens": "คะน้าฝรั่ง (Collard Greens)",
            "Endive": "เอนไดฟ์ (Endive)"
        };

        fields.forEach(field => {
            const thaiPlantName = plantNameMap[field.plant_type] || field.plant_type;

            // สร้างการ์ดแสดงแปลง
            const col = document.createElement('div');
            col.className = 'col-md-6';
            col.innerHTML = `
                <div class="card field-card shadow-sm border-success border-opacity-25">
                    <div class="card-body d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1 text-success fw-bold">${field.name}</h6>
                            <small class="text-muted">พืช: ${thaiPlantName}</small>
                        </div>
                        <button class="btn btn-sm btn-outline-primary" onclick="jumpToPredict(${field.id})">
                            ตรวจสุขภาพ ➔
                        </button>
                    </div>
                </div>
            `;
            grid.appendChild(col);

            // เติมเข้า Dropdown
            selPredict.innerHTML += `<option value="${field.id}">${field.name} (${thaiPlantName})</option>`;
            selHistory.innerHTML += `<option value="${field.id}">${field.name} (${thaiPlantName})</option>`;
        });

        // Render lucide icons for dynamically added elements
        lucide.createIcons();
    } catch (error) {
        console.error('Error loading fields:', error);
    }
}

document.getElementById('addFieldForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/fields/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: document.getElementById('fieldName').value,
                plant_type: document.getElementById('plantType').value
            })
        });

        if (response.ok) {
            const newField = await response.json();
            document.getElementById('addFieldForm').reset();
            await loadFields();
            // วาร์ปไปหน้าทำนายอัตโนมัติ พร้อมล็อกเป้าแปลงใหม่
            jumpToPredict(newField.id);
        } else {
            alert('❌ เกิดข้อผิดพลาดในการเพิ่มแปลง');
        }
    } catch (error) {
        console.error('Error adding field:', error);
    } finally {
        btn.disabled = false;
    }
});

// ==========================================
// 3. ระบบพยากรณ์สุขภาพ (Analyze)
// ==========================================
// Toggle Pro Mode
document.getElementById('proModeToggle').addEventListener('change', (e) => {
    const isPro = e.target.checked;
    const simpleDiv = document.getElementById('simpleModeInputs');
    const advDiv = document.getElementById('advancedModeInputs');

    if (isPro) {
        simpleDiv.classList.add('d-none');
        advDiv.classList.remove('d-none');
        document.getElementById('advSoilMoisture').required = true;
        document.getElementById('advNitrogenLevel').required = true;
        document.getElementById('soilMoisture').required = false;
        document.getElementById('nitrogenLevel').required = false;
    } else {
        simpleDiv.classList.remove('d-none');
        advDiv.classList.add('d-none');
        document.getElementById('advSoilMoisture').required = false;
        document.getElementById('advNitrogenLevel').required = false;
        document.getElementById('soilMoisture').required = true;
        document.getElementById('nitrogenLevel').required = true;
    }
});

document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fieldId = document.getElementById('fieldSelectPredict').value;
    const btn = e.target.querySelector('button');

    // ดึงค่าตามโหมด (Simple หรือ Pro)
    const isPro = document.getElementById('proModeToggle').checked;
    const soil_val = isPro ? parseFloat(document.getElementById('advSoilMoisture').value) : parseFloat(document.getElementById('soilMoisture').value);
    const nitro_val = isPro ? parseFloat(document.getElementById('advNitrogenLevel').value) : parseFloat(document.getElementById('nitrogenLevel').value);

    // รวบรวมข้อมูล! ส่งค่าที่หน้าจอถามแค่ 2 ค่า ส่วนที่เหลือยัด 0.0 ให้ระบบเก่าไม่แตก
    const payload = {
        field_id: parseInt(fieldId),
        soil_moisture: soil_val,
        nitrogen_level: nitro_val,

        // ค่าโมเดลไม่ใช้ (ดัมมี่ให้ Schema ของ FastAPI พึงพอใจ)
        ambient_temperature: 0.0,
        soil_temperature: 0.0,
        humidity: 0.0,
        light_intensity: 0.0,
        soil_ph: 0.0,
        phosphorus_level: 0.0,
        potassium_level: 0.0,
        chlorophyll_content: 0.0,
        electrochemical_signal: 0.0
    };

    btn.innerHTML = '⏳ กำลังประมวลผล...';
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/predictions/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const result = await response.json();
            showResult(result, soil_val, nitro_val);
            // อัปเดต Dropdown ทันทีในแท็บ History ป้องกันผู้ใช้สลับไปแล้วงง
            document.getElementById('fieldSelectHistory').value = fieldId;
        } else {
            const err = await response.json();
            alert(`❌ Error: ${err.detail}`);
        }
    } catch (error) {
        console.error('Prediction Error:', error);
        alert('❌ ไม่สามารถเชื่อมต่อระบบวิเคราะห์ได้');
    } finally {
        btn.innerHTML = '🧠 วิเคราะห์สุขภาพทันที';
        btn.disabled = false;
    }
});

function showResult(result, soil_val, nitro_val) {
    const area = document.getElementById('resultArea');
    const badge = document.getElementById('predictResultBadge');

    area.classList.remove('d-none');

    const thaiStatusMap = {
        0: "ปกติ (Healthy)",
        1: "พืชเริ่มเครียด (Moderate Stress)",
        2: "เครียดสูง (High Stress)"
    };
    const thaiStatus = thaiStatusMap[result.predicted_code] || result.predicted_status;

    // ตกแต่งป้ายผลลัพธ์
    badge.textContent = thaiStatus;
    badge.className = 'p-3 rounded-pill text-white fw-bold d-inline-block px-5 fs-4 shadow-sm mb-3 ';

    // --- Actionable Insights Logic ---
    const suggestionBox = document.getElementById('suggestionBox');
    const suggestionList = document.getElementById('suggestionList');
    suggestionList.innerHTML = '';

    if (result.predicted_code === 0) {
        badge.classList.add('bg-success');
        badge.innerHTML = `<i data-lucide="check-circle" class="me-2 text-white"></i> ${thaiStatus}`;
        suggestionBox.style.borderLeftColor = '#15803D';
        suggestionBox.style.backgroundColor = '#F0FDF4';
    } else if (result.predicted_code === 1) {
        badge.classList.add('bg-warning', 'text-dark');
        badge.innerHTML = `<i data-lucide="alert-triangle" class="me-2 text-dark"></i> ${thaiStatus}`;
        suggestionBox.style.borderLeftColor = '#EAB308';
        suggestionBox.style.backgroundColor = '#FEFCE8';
    } else {
        badge.classList.add('bg-danger');
        badge.innerHTML = `<i data-lucide="alert-octagon" class="me-2 text-white"></i> ${thaiStatus}`;
        suggestionBox.style.borderLeftColor = '#DC2626';
        suggestionBox.style.backgroundColor = '#FEF2F2';
    }

    // สร้างคำแนะนำตามข้อมูลที่รับมา
    // 0. ดูสถานะความเครียดภาพรวม (Overall Status)
    if (result.predicted_code === 1) { // Moderate Stress
        suggestionList.innerHTML += `<li class="mb-3"><span class="fs-5 me-2" style="color:#d97706">🔍</span> <strong style="color:#b45309">พบความเครียดระดับกลาง:</strong> ควรเริ่มปรับปรุงสภาพการดูแล และติดตามอาการอย่างใกล้ชิด</li>`;
    } else if (result.predicted_code === 2) { // High Stress
        suggestionList.innerHTML += `<li class="mb-3"><span class="fs-5 me-2">🚨</span> <strong class="text-danger">วิกฤตความเครียดสูง:</strong> ต้องดำเนินการจัดการปัจจัยเสี่ยงด้านล่างทันที เพื่อไม่ให้พืชตาย</li>`;
    }

    // 1. ความชื้นในดิน (Soil Moisture)
    if (soil_val < 20) {
        suggestionList.innerHTML += `<li class="mb-3"><span class="fs-5 me-2">💧</span> <strong class="text-danger">ดินแห้งเกินไป:</strong> ควรรดน้ำทันทีและเพิ่มความถี่ในการให้น้ำ</li>`;
    } else if (soil_val >= 30) {
        suggestionList.innerHTML += `<li class="mb-3"><span class="fs-5 me-2">⚠️</span> <strong class="text-warning text-dark">ดินแฉะ/น้ำขัง:</strong> ควรงดรดน้ำชั่วคราว เช็คระบบระบายน้ำเพื่อกันรากเน่า</li>`;
    } else {
        suggestionList.innerHTML += `<li class="mb-3"><span class="fs-5 me-2">💧</span> <strong class="text-success">ความชื้นกำลังดี:</strong> รักษาระดับการรดน้ำตามปกติไว้</li>`;
    }

    // 2. ระดับไนโตรเจน (Nitrogen Level)
    if (nitro_val <= 19.5) {
        suggestionList.innerHTML += `<li><span class="fs-5 me-2">🌿</span> <strong class="text-warning text-dark">ไนโตรเจนต่ำ (ใบเหลือง):</strong> ควรเสริมปุ๋ยไนโตรเจน (N) เช่น 46-0-0 เพื่อเร่งเขียว</li>`;
    } else if (nitro_val >= 21) {
        suggestionList.innerHTML += `<li><span class="fs-5 me-2">🛑</span> <strong class="text-danger">ไนโตรเจนสูงเกิน:</strong> หยุดให้ปุ๋ยเคมี(N) ชั่วคราว ป้องกันพืชอวบน้ำอ่อนแอต่อโรค</li>`;
    } else {
        suggestionList.innerHTML += `<li><span class="fs-5 me-2">🌿</span> <strong class="text-success">ธาตุอาหารสมดุล:</strong> ใบสีเขียวปกติ ไม่จำเป็นต้องใส่ปุ๋ยเคมีเพิ่ม</li>`;
    }

    suggestionBox.classList.remove('d-none');
    lucide.createIcons();
}

// ==========================================
// 4. ระบบประวัติ (History Tracker)
// ==========================================
function refreshHistory() {
    const id = document.getElementById('fieldSelectHistory').value;
    if (id) loadHistory(id);
}

async function loadHistory(fieldId) {
    const tbody = document.getElementById('historyTableBody');
    if (!fieldId) return;

    tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4">กำลังโหลดสมุดพก...</td></tr>';

    try {
        const response = await fetch(`${API_BASE_URL}/fields/${fieldId}`);
        const data = await response.json();
        const history = data.predictions;

        if (!history || history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-4">แปลงนี้ยังไม่เคยวิเคราะห์สุขภาพ</td></tr>';
            return;
        }

        tbody.innerHTML = '';

        // เรียงจากการบันทึกล่าสุดขึ้นก่อน
        history.reverse().forEach(row => {
            const dateStr = new Date(row.created_at).toLocaleString('th-TH', {
                day: '2-digit', month: 'short', year: 'numeric',
                hour: '2-digit', minute: '2-digit'
            });

            // ดึงค่ามาแปลงเป็นคำพูดให้ชื่นใจ พร้อมแสดงตัวเลขในวงเล็บ (Pro Mode support)
            const soilLabel = row.soil_moisture <= 15 ? "แห้ง" : (row.soil_moisture >= 35 ? "แฉะ/ขัง" : "ชื้นปกติ");
            const nitLabel = row.nitrogen_level <= 18 ? "เหลืองซีด" : (row.nitrogen_level >= 30 ? "เขียวเข้ม" : "เขียวปกติ");

            const soilDisp = `${soilLabel} <small class="text-secondary font-monospace fw-bold ms-1" style="font-size:0.75rem;">(${Number(row.soil_moisture).toFixed(2)})</small>`;
            const nitDisp = `${nitLabel} <small class="text-secondary font-monospace fw-bold ms-1" style="font-size:0.75rem;">(${Number(row.nitrogen_level).toFixed(2)})</small>`;

            let color = 'secondary';
            if (row.predicted_code === 0) color = 'success';
            else if (row.predicted_code === 1) color = 'warning text-dark';
            else if (row.predicted_code === 2) color = 'danger';

            const thaiStatusMap = {
                0: "ปกติ (Healthy)",
                1: "พืชเริ่มเครียด (Moderate Stress)",
                2: "เครียดสูง (High Stress)"
            };
            const thaiStatus = thaiStatusMap[row.predicted_code] || row.predicted_status;

            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid #f1f5f9';
            tr.innerHTML = `
                <td class="small text-muted font-monospace">${dateStr}</td>
                <td><span class="badge" style="background-color:#F8FAFC; color:#475569; border: 1px solid #E2E8F0;"><i data-lucide="droplet" style="width:12px; height:12px;" class="me-1"></i> ${soilDisp}</span></td>
                <td><span class="badge" style="background-color:#F8FAFC; color:#475569; border: 1px solid #E2E8F0;"><i data-lucide="leaf" style="width:12px; height:12px;" class="me-1"></i> ${nitDisp}</span></td>
                <td><span class="badge bg-${color} px-3 py-2 fw-bold shadow-sm">${thaiStatus}</span></td>
            `;
            tbody.appendChild(tr);
        });

        lucide.createIcons();
    } catch (error) {
        console.error('Error loading history:', error);
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">ไม่สามารถโหลดข้อมูลได้</td></tr>';
    }
}